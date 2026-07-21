/// Variable ID — index into the VM's variable store.
pub type VarId = u32;
/// Block ID — index into the VM's block array.
pub type BlockId = u32;

use rustc_hash::{FxHashMap, FxHasher};
use serde::de::{Error as _, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Index;
use std::sync::Arc;

/// Number of variable names held by one allocation. A `Box<str>` is 16 bytes
/// on the supported 64-bit target, so one full chunk is 256 KiB. This avoids the
/// hundreds-of-MiB copy-on-growth cliff of a single `Vec<String>` while
/// preserving exact `VarId` indexing.
const NAME_CHUNK_SHIFT: usize = 14;
const NAME_CHUNK_LEN: usize = 1 << NAME_CHUNK_SHIFT;
const NAME_CHUNK_MASK: usize = NAME_CHUNK_LEN - 1;

/// Chunked variable-name sequence indexed by `VarId`.
///
/// Its custom serde implementation intentionally presents one ordinary flat
/// sequence, byte-compatible with the former `Vec<String>` representation in
/// bincode `.swcp` files. Chunking is an in-memory allocation strategy only.
#[derive(Debug, Clone, Default)]
pub struct NameTable {
    chunks: Vec<Vec<Box<str>>>,
    len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameTableOverflow;

impl fmt::Display for NameTableOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("variable-name count exceeds the u32 VarId space")
    }
}

impl std::error::Error for NameTableOverflow {}

fn checked_name_var_id(len: usize) -> Result<VarId, NameTableOverflow> {
    let id = VarId::try_from(len).map_err(|_| NameTableOverflow)?;
    // `num_vars` is also a u32, so admitting id == u32::MAX would make the
    // post-push count unrepresentable even though that final id itself fits.
    if id == VarId::MAX {
        return Err(NameTableOverflow);
    }
    Ok(id)
}

fn checked_name_count(count: usize) -> Result<(), NameTableOverflow> {
    if count > VarId::MAX as usize {
        return Err(NameTableOverflow);
    }
    Ok(())
}

impl NameTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn try_push(&mut self, name: Box<str>) -> Result<VarId, NameTableOverflow> {
        let id = checked_name_var_id(self.len)?;
        if self
            .chunks
            .last()
            .map_or(true, |chunk| chunk.len() == NAME_CHUNK_LEN)
        {
            self.chunks.push(Vec::with_capacity(NAME_CHUNK_LEN));
        }
        self.chunks
            .last_mut()
            .expect("name chunk created above")
            .push(name);
        self.len += 1;
        Ok(id)
    }

    pub fn get(&self, index: usize) -> Option<&str> {
        if index >= self.len {
            return None;
        }
        self.chunks[index >> NAME_CHUNK_SHIFT]
            .get(index & NAME_CHUNK_MASK)
            .map(Box::as_ref)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.chunks
            .iter()
            .flat_map(|chunk| chunk.iter().map(Box::as_ref))
    }
}

impl From<Vec<String>> for NameTable {
    fn from(names: Vec<String>) -> Self {
        let mut table = Self::new();
        for name in names {
            table
                .try_push(name.into_boxed_str())
                .expect("Vec length admitted by the u32 VarId space");
        }
        table
    }
}

impl Index<usize> for NameTable {
    type Output = str;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap_or_else(|| {
            panic!(
                "name-table index {index} out of bounds for {} names",
                self.len
            )
        })
    }
}

impl Serialize for NameTable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len))?;
        for name in self.iter() {
            seq.serialize_element(name)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for NameTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NameTableVisitor;

        impl<'de> Visitor<'de> for NameTableVisitor {
            type Value = NameTable;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a flat sequence of variable names")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                if let Some(count) = seq.size_hint() {
                    checked_name_count(count).map_err(A::Error::custom)?;
                }
                let mut table = NameTable::new();
                while let Some(name) = seq.next_element::<Box<str>>()? {
                    table.try_push(name).map_err(A::Error::custom)?;
                }
                Ok(table)
            }
        }

        deserializer.deserialize_seq(NameTableVisitor)
    }
}

/// Collision-checked compact name lookup. Keys remain in `NameTable`; this
/// index stores only a 64-bit hash and `VarId`, avoiding a second allocation
/// and owned copy of every deeply inlined name.
#[derive(Debug, Default)]
pub struct NameIndex {
    first_by_hash: FxHashMap<u64, VarId>,
    collisions: FxHashMap<u64, Vec<VarId>>,
    entries: usize,
}

impl NameIndex {
    pub fn from_names(names: &NameTable) -> Self {
        let mut index = Self {
            first_by_hash: FxHashMap::with_capacity_and_hasher(names.len(), Default::default()),
            collisions: FxHashMap::default(),
            entries: 0,
        };
        for offset in 0..names.len() {
            let id = checked_name_var_id(offset)
                .expect("NameTable deserialization admitted every VarId");
            index.insert(names, id);
        }
        index
    }

    pub fn lookup(&self, names: &NameTable, name: &str) -> Option<VarId> {
        self.lookup_hashed(names, name, hash_name(name))
    }

    pub fn insert(&mut self, names: &NameTable, id: VarId) {
        let name = &names[id as usize];
        self.insert_hashed(names, name, id, hash_name(name));
    }

    /// Remove one exact name/id entry from the lowering index.
    ///
    /// File-package lowering can discard completed inline-frame lookups once
    /// their numeric `VarId`s have been embedded in bytecode. The names remain
    /// in `NameTable` for diagnostics and wire compatibility; only this derived
    /// hash index entry is released. Hash collisions stay exact by promoting a
    /// colliding id when the primary slot is removed.
    pub fn remove(&mut self, names: &NameTable, id: VarId) -> bool {
        let Some(name) = names.get(id as usize) else {
            return false;
        };
        self.remove_hashed(id, hash_name(name))
    }

    pub fn len(&self) -> usize {
        self.entries
    }

    fn lookup_hashed(&self, names: &NameTable, name: &str, hash: u64) -> Option<VarId> {
        let first = *self.first_by_hash.get(&hash)?;
        if &names[first as usize] == name {
            return Some(first);
        }
        self.collisions
            .get(&hash)
            .and_then(|ids| ids.iter().copied().find(|id| &names[*id as usize] == name))
    }

    fn insert_hashed(&mut self, names: &NameTable, name: &str, id: VarId, hash: u64) {
        let Some(first) = self.first_by_hash.get_mut(&hash) else {
            self.first_by_hash.insert(hash, id);
            self.entries += 1;
            return;
        };
        if &names[*first as usize] == name {
            *first = id;
            return;
        }

        let collisions = self.collisions.entry(hash).or_default();
        if let Some(existing) = collisions
            .iter_mut()
            .find(|existing| &names[**existing as usize] == name)
        {
            *existing = id;
            return;
        }
        collisions.push(id);
        self.entries += 1;
    }

    fn remove_hashed(&mut self, id: VarId, hash: u64) -> bool {
        let Some(first) = self.first_by_hash.get(&hash).copied() else {
            return false;
        };

        if first == id {
            let replacement = self.collisions.get_mut(&hash).and_then(|ids| ids.pop());
            if let Some(replacement) = replacement {
                self.first_by_hash.insert(hash, replacement);
                if self.collisions.get(&hash).is_some_and(|ids| ids.is_empty()) {
                    self.collisions.remove(&hash);
                }
            } else {
                self.first_by_hash.remove(&hash);
            }
            self.entries -= 1;
            return true;
        }

        let Some(collisions) = self.collisions.get_mut(&hash) else {
            return false;
        };
        let Some(position) = collisions.iter().position(|candidate| *candidate == id) else {
            return false;
        };
        collisions.swap_remove(position);
        if collisions.is_empty() {
            self.collisions.remove(&hash);
        }
        self.entries -= 1;
        true
    }
}

fn hash_name(name: &str) -> u64 {
    let mut hasher = FxHasher::default();
    name.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod name_table_tests {
    use super::{
        checked_name_count, checked_name_var_id, Block, BlockId, CompiledProgram, Expr, MemMapInfo,
        NameIndex, NameTable, SemanticsMode, VarId, NAME_CHUNK_LEN,
    };
    use rustc_hash::FxHashMap;
    use std::sync::Arc;

    #[test]
    fn indexes_exactly_across_chunk_boundaries() {
        let mut table = NameTable::new();
        for index in 0..(NAME_CHUNK_LEN + 2) {
            assert_eq!(
                table.try_push(format!("$v{index}").into_boxed_str()),
                Ok(index as VarId)
            );
        }

        assert_eq!(table.chunks.len(), 2);
        assert_eq!(table.chunks[0].capacity(), NAME_CHUNK_LEN);
        assert!(NAME_CHUNK_LEN * std::mem::size_of::<Box<str>>() <= 256 * 1024);
        assert_eq!(
            table[NAME_CHUNK_LEN - 1],
            format!("$v{}", NAME_CHUNK_LEN - 1)
        );
        assert_eq!(table[NAME_CHUNK_LEN], format!("$v{NAME_CHUNK_LEN}"));
        assert_eq!(
            table[NAME_CHUNK_LEN + 1],
            format!("$v{}", NAME_CHUNK_LEN + 1)
        );
        assert_eq!(table.iter().count(), NAME_CHUNK_LEN + 2);
    }

    #[test]
    fn bincode_wire_matches_legacy_flat_vec_in_both_directions() {
        let legacy = Arc::new(vec![
            "$x".to_string(),
            "inline$callee$17$$value".to_string(),
            "$M.0.shadow".to_string(),
        ]);
        let table = Arc::new(NameTable::from(legacy.as_ref().clone()));

        let legacy_wire = bincode::serialize(&legacy).unwrap();
        let table_wire = bincode::serialize(&table).unwrap();
        assert_eq!(table_wire, legacy_wire);

        let decoded_table: Arc<NameTable> = bincode::deserialize(&legacy_wire).unwrap();
        assert_eq!(
            decoded_table.iter().map(str::to_string).collect::<Vec<_>>(),
            legacy.as_ref().clone()
        );
        let decoded_legacy: Arc<Vec<String>> = bincode::deserialize(&table_wire).unwrap();
        assert_eq!(decoded_legacy, legacy);
    }

    #[test]
    fn rejects_unrepresentable_var_count_before_mutation() {
        assert_eq!(checked_name_var_id(0), Ok(0));
        assert_eq!(
            checked_name_var_id((VarId::MAX - 1) as usize),
            Ok(VarId::MAX - 1)
        );
        assert!(checked_name_var_id(VarId::MAX as usize).is_err());
        assert!(checked_name_count(VarId::MAX as usize).is_ok());
        if usize::BITS > 32 {
            assert!(checked_name_var_id(VarId::MAX as usize + 1).is_err());
            assert!(checked_name_count(VarId::MAX as usize + 1).is_err());
        }
    }

    #[test]
    fn compact_index_resolves_forced_hash_collisions_by_full_name() {
        let mut names = NameTable::new();
        let first = names.try_push("$first".into()).unwrap();
        let second = names.try_push("$second".into()).unwrap();
        let third = names.try_push("$third".into()).unwrap();
        let forced_hash = 7;
        let mut index = NameIndex::default();
        index.insert_hashed(&names, "$first", first, forced_hash);
        index.insert_hashed(&names, "$second", second, forced_hash);
        index.insert_hashed(&names, "$third", third, forced_hash);

        assert_eq!(
            index.lookup_hashed(&names, "$first", forced_hash),
            Some(first)
        );
        assert_eq!(
            index.lookup_hashed(&names, "$second", forced_hash),
            Some(second)
        );
        assert_eq!(
            index.lookup_hashed(&names, "$third", forced_hash),
            Some(third)
        );
        assert_eq!(index.lookup_hashed(&names, "$missing", forced_hash), None);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn compact_index_removal_preserves_forced_hash_collisions() {
        let mut names = NameTable::new();
        let first = names.try_push("$first".into()).unwrap();
        let second = names.try_push("$second".into()).unwrap();
        let third = names.try_push("$third".into()).unwrap();
        let forced_hash = 7;
        let mut index = NameIndex::default();
        index.insert_hashed(&names, "$first", first, forced_hash);
        index.insert_hashed(&names, "$second", second, forced_hash);
        index.insert_hashed(&names, "$third", third, forced_hash);

        assert!(index.remove_hashed(first, forced_hash));
        assert_eq!(index.lookup_hashed(&names, "$first", forced_hash), None);
        assert_eq!(
            index.lookup_hashed(&names, "$second", forced_hash),
            Some(second)
        );
        assert_eq!(
            index.lookup_hashed(&names, "$third", forced_hash),
            Some(third)
        );

        assert!(index.remove_hashed(second, forced_hash));
        assert_eq!(
            index.lookup_hashed(&names, "$third", forced_hash),
            Some(third)
        );
        assert!(!index.remove_hashed(second, forced_hash));
        assert!(index.remove_hashed(third, forced_hash));
        assert_eq!(index.lookup_hashed(&names, "$third", forced_hash), None);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn deserialized_program_rebuilds_compact_index_for_exact_input_lookup() {
        let names = NameTable::from(vec![
            "$public".to_string(),
            "$secret".to_string(),
            "$secret.shadow".to_string(),
        ]);
        let program = CompiledProgram {
            blocks: vec![],
            label_to_block: FxHashMap::default(),
            name_index: NameIndex::from_names(&names),
            var_names: Arc::new(names),
            entry_block: 0,
            entry_preconditions: vec![],
            mem_maps: vec![],
            num_vars: 3,
            curr_addr_id: None,
            curr_addr_shadow_id: None,
            m0_id: None,
            m0_shadow_id: None,
            loop_header_live_vars: FxHashMap::default(),
            is_loop_header: vec![],
            block_innermost_header: vec![],
            loop_parent_header: vec![],
            static_scalars: vec![],
            mode: SemanticsMode::Bv,
        };

        let wire = bincode::serialize(&program).unwrap();
        let legacy_wire = bincode::serialize(&(
            Vec::<Block>::new(),
            Arc::new(vec![
                "$public".to_string(),
                "$secret".to_string(),
                "$secret.shadow".to_string(),
            ]),
            0u32,
            Vec::<Expr>::new(),
            Vec::<MemMapInfo>::new(),
            3u32,
            None::<VarId>,
            None::<VarId>,
            None::<VarId>,
            None::<VarId>,
            FxHashMap::<BlockId, Vec<VarId>>::default(),
            Vec::<bool>::new(),
            Vec::<Option<BlockId>>::new(),
            Vec::<Option<BlockId>>::new(),
            Vec::<(VarId, i64)>::new(),
            SemanticsMode::Bv,
        ))
        .unwrap();
        assert_eq!(
            wire, legacy_wire,
            "full package wire remains legacy-compatible"
        );

        let mut loaded: CompiledProgram = bincode::deserialize(&legacy_wire).unwrap();
        assert_eq!(loaded.name_index.len(), 0, "runtime index is serde-skipped");
        loaded.rebuild_runtime_name_index();
        assert_eq!(loaded.name_index.len(), 3);
        assert_eq!(loaded.lookup_var("$public"), Some(0));
        assert_eq!(loaded.lookup_var("$secret"), Some(1));
        assert_eq!(loaded.lookup_var("$secret.shadow"), Some(2));
        assert_eq!(loaded.lookup_var("$missing"), None);
    }
}

/// A compiled expression — tree of Rust enums, no Python objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    /// Variable lookup by interned ID
    Var(VarId),
    /// Integer constant
    Const(i64),
    /// Boolean constant
    Bool(bool),
    /// Binary operator (Boogie-level: ==, !=, &&, ||, etc.)
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Builtin function call (e.g. $add.i32, $sext.i32.i64)
    Builtin { fn_id: BuiltinFn, args: Vec<Expr> },
    /// Memory store: $store.iN(map, index, value). A zero width denotes one
    /// direct Boogie map element and is resolved from the map at execution.
    Store {
        bit_width: u8,
        map: Box<Expr>,
        index: Box<Expr>,
        value: Box<Expr>,
    },
    /// Memory load: $load.iN(map, index). A zero width denotes one direct
    /// Boogie map element and is resolved from the map at execution.
    Load {
        bit_width: u8,
        map: Box<Expr>,
        index: Box<Expr>,
    },
    /// if cond then t else e
    IfThenElse {
        cond: Box<Expr>,
        then_: Box<Expr>,
        else_: Box<Expr>,
    },
    /// Logical negation: !expr
    Not(Box<Expr>),
    /// $isExternal — always returns 0
    IsExternal,
    /// Integer constant beyond i64 — Int (exact-ℤ) mode only. BV mode never
    /// constructs this (an out-of-u64 literal is a loud lowering error there).
    /// Kept as the LAST variant: bincode `.swcp` encoding is positional, so
    /// appending preserves deserialization of every pre-existing package.
    ConstBig(Box<num_bigint::BigInt>),
}

/// Binary operators at the Boogie level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Implies,
    Iff,
    Sub,
    Mul,
    Add,
    // Keep new variants at the end: bincode encodes enum variants by their
    // positional discriminant, so appending preserves existing `.swcp`
    // packages.
    Div,
    Mod,
}

/// All builtin functions from generate_function_map + fn_map_to_op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinFn {
    // Arithmetic
    Add { bits: u8 },
    Sub { bits: u8 },
    Mul { bits: u8 },

    // Bitwise
    And { bits: u8 },
    Or { bits: u8 },
    Xor { bits: u8 },
    Not { bits: u8 },

    // Shifts
    Shl { bits: u8 },
    Lshr { bits: u8 },
    Ashr { bits: u8 },

    // Signed comparisons
    Slt { bits: u8 },
    Sle { bits: u8 },
    Sgt { bits: u8 },
    Sge { bits: u8 },

    // Unsigned comparisons
    Ult { bits: u8 },
    Ule { bits: u8 },
    Ugt { bits: u8 },
    Uge { bits: u8 },

    // Equality
    BvEq { bits: u8 },
    BvNe { bits: u8 },

    // Division / remainder
    Udiv { bits: u8 },
    Sdiv { bits: u8 },
    Urem { bits: u8 },
    Srem { bits: u8 },

    // Casts
    Sext { src: u8, dst: u8 },
    Zext { src: u8, dst: u8 },
    Trunc { dst: u8 },
    Bitcast,
    P2i,
    I2p,

    // Boolean result variants (for .ref.bool functions)
    SltBool { bits: u8 },
    SleBool { bits: u8 },
    SgtBool { bits: u8 },
    SgeBool { bits: u8 },

    // SMT-LIB integer division / modulus — the RESIDUAL div/rem intrinsics of
    // the SMACK integer encoding ({:builtin "div"} $idiv.iN / {:builtin "mod"}
    // $smod.iN). Int mode only; BV-mode programs never name them. Appended
    // last: bincode `.swcp` variant indices are positional.
    Idiv { bits: u8 },
    Smod { bits: u8 },
}

/// A compiled statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    /// `if (cond) { then_body } else { else_body }` — structured
    /// branching emitted by the diffprod corerel reify path for
    /// `IfRel`-inside-loop and similar shapes. Body vectors must
    /// NOT contain Goto/Return — bpl_emit honors this; lowering
    /// asserts.
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    /// `while (cond) { body }` — structured loop emitted by
    /// diffprod when corerel reify encounters a nested loop inside
    /// another structured construct. Body must NOT contain
    /// Goto/Return; lowering debug-asserts.
    While { cond: Expr, body: Vec<Stmt> },
    /// x := expr (single assignment, most common)
    Assign1 { lhs: VarId, rhs: Expr },
    /// x, y := e1, e2 (multi-assignment)
    AssignN { lhs: Vec<VarId>, rhs: Vec<Expr> },
    /// assert expr
    Assert { expr: Expr },
    /// assume expr (non-quantified, non-trivial)
    Assume { expr: Expr },
    /// assume true — skip
    AssumeTrue,
    /// Loop header snapshot: record current values of live vars for trace
    LoopHeaderSnap { live_vars: Vec<VarId> },
    /// havoc x, y, ...
    Havoc { vars: Vec<VarId> },
    /// havoc $CurrAddr (or .shadow) — allocation: read size from alloc_size_var,
    /// compute new_addr = (old_addr + size + 255) & ~255, set, trace, then clear.
    HavocCurrAddr {
        var_id: VarId,
        alloc_size_var: VarId,
    },
    /// goto label
    Goto { targets: Vec<BlockId> },
    /// return
    Return,
    /// Calls that are ignored (verifier_nondet, etc.)
    CallIgnored,
    /// __VERIFIER_nondet_* / __SMACK_nondet_* with assignment results.
    CallNondet { assignments: Vec<VarId> },
    /// call printf.ref.* — read format string from $M.0, format args, print
    CallPrintf { args: Vec<Expr> },
    /// call time.cross_product
    CallTime {
        assignments: Vec<VarId>,
        args: Vec<Expr>,
    },
    /// call write.cross_product
    CallWrite {
        assignments: Vec<VarId>,
        args: Vec<Expr>,
    },
    /// call read.cross_product
    CallRead { args: Vec<Expr> },
    /// call llvm.memmove/llvm.memcpy intrinsic over byte-addressed memory maps
    CallMemmove { args: Vec<Expr> },
    /// Quantified assume for memset (&&)
    QuantMemsetWrite {
        m_ret: VarId,
        dst: VarId,
        len: VarId,
        val: VarId,
    },
    /// Quantified assume for memset preserve (<)
    QuantMemsetPreserveLt {
        m_ret: VarId,
        m_src: VarId,
        dst: VarId,
    },
    /// Quantified assume for memset preserve (>=)
    QuantMemsetPreserveGe {
        m_ret: VarId,
        m_src: VarId,
        dst: VarId,
        len: VarId,
    },
    /// Quantified assume for memcpy (&&)
    QuantMemcpyWrite {
        m_ret: VarId,
        m_src: VarId,
        dst: VarId,
        src: VarId,
        len: VarId,
    },
    /// Quantified assume for memcpy preserve (<)
    QuantMemcpyPreserveLt {
        m_ret: VarId,
        m_src: VarId,
        dst: VarId,
    },
    /// Quantified assume for memcpy preserve (>=)
    QuantMemcpyPreserveGe {
        m_ret: VarId,
        m_src: VarId,
        dst: VarId,
        len: VarId,
    },
    /// Interpreter-internal lifetime boundary for map-typed variables in one
    /// native-inliner frame. The inliner emits this only after every callee
    /// return value has been copied into its caller.
    ///
    /// This is maintenance metadata, not a Boogie statement: it owns no
    /// verifier PC, trace event, or execution-budget step. Keep this variant
    /// appended so existing bincode `.swcp` statement discriminants remain
    /// stable.
    ReleaseMaps { vars: Vec<VarId> },
}

impl Stmt {
    /// Whether this statement is native runtime maintenance rather than a
    /// verifier-addressable source statement.
    #[inline]
    pub fn is_internal_maintenance(&self) -> bool {
        matches!(self, Self::ReleaseMaps { .. })
    }
}

/// A compiled block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub name: String,
    pub id: BlockId,
    /// Statements *except* the last one (which is always goto/return)
    pub body: Vec<Stmt>,
    /// The terminator (goto or return)
    pub terminator: Stmt,
    /// PC of the first statement in this block
    pub start_pc: u32,
    /// The assume condition for branch resolution (first assume in the block)
    pub assume_cond: Option<Expr>,
}

/// Metadata for a memory map variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemMapInfo {
    pub name: String,
    pub var_id: VarId,
    pub index_bit_width: u8,
    pub element_bit_width: u8,
}

/// Arithmetic semantics the program was compiled under.
///
/// SMACK emits the same `type i32 = int` alias in both encodings; the
/// truthful signal is the prelude body shape (see Python
/// `interpreter.utils.integer_encoding.detect_integer_encoding`):
///
/// * `Int` — SMACK unbounded-integer encoding: prelude bodies are plain
///   `(i1 + i2)` over mathematical integers. Wrapping/masking is WRONG here
///   (the c2i_094 `j=(n+1)(n+2)/2` false refutation at n=65535 was 32-bit
///   wrap applied to a math-int program).
/// * `Bv` — SMACK bit-vector encoding (`$add.bv32` intrinsics): the wrapping
///   algebra implemented by `builtins.rs` / `vm.rs` is the correct model.
///
/// Step 1 (mode plumbing) only CARRIES this tag; the VM still evaluates with
/// wrapping semantics in both modes. Step 3 switches evaluation on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticsMode {
    /// SMACK unbounded-integer encoding (`type i32 = int`, plain `(i1+i2)` prelude).
    Int,
    /// SMACK bit-vector encoding — wrapping algebra is the correct model.
    Bv,
}

impl SemanticsMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            SemanticsMode::Int => "int",
            SemanticsMode::Bv => "bv",
        }
    }

    pub fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "int" => Ok(SemanticsMode::Int),
            "bv" => Ok(SemanticsMode::Bv),
            other => Err(format!(
                "unknown semantics mode {:?} (expected \"int\" or \"bv\")",
                other
            )),
        }
    }
}

/// A compiled program ready for VM execution.
#[derive(Debug, Serialize, Deserialize)]
pub struct CompiledProgram {
    /// All blocks, indexed by BlockId
    pub blocks: Vec<Block>,
    /// Block name → BlockId (derived; rebuilt on load via `rebuild_lookup_maps`, not serialized)
    #[serde(skip)]
    pub label_to_block: rustc_hash::FxHashMap<String, BlockId>,
    /// Variable name list indexed by VarId
    pub var_names: Arc<NameTable>,
    /// Compact variable-name index (derived; rebuilt on load, not serialized).
    /// Full names remain owned only by `var_names`.
    #[serde(skip)]
    pub name_index: NameIndex,
    /// Entry block ID
    pub entry_block: BlockId,
    /// Preconditions that must hold before the entry block executes.
    pub entry_preconditions: Vec<Expr>,
    /// Memory map info for each VarId that holds a memory map
    pub mem_maps: Vec<MemMapInfo>,
    /// Total number of variables
    pub num_vars: u32,
    /// VarId for $CurrAddr
    pub curr_addr_id: Option<VarId>,
    /// VarId for $CurrAddr.shadow
    pub curr_addr_shadow_id: Option<VarId>,
    /// VarId for $M.0
    pub m0_id: Option<VarId>,
    /// VarId for $M.0.shadow
    pub m0_shadow_id: Option<VarId>,
    /// Loop header block → live variable IDs to snapshot
    pub loop_header_live_vars: rustc_hash::FxHashMap<BlockId, Vec<VarId>>,
    /// BlockId → true iff this block is a loop header.  Parallel arrays
    /// to `blocks`; len == blocks.len().
    pub is_loop_header: Vec<bool>,
    /// BlockId → innermost enclosing loop header's BlockId (or None if
    /// this block is not inside any loop).
    pub block_innermost_header: Vec<Option<BlockId>>,
    /// For each loop header BlockId, its parent (immediately-enclosing)
    /// loop header, or None if it is a top-level loop.  Indexed by the
    /// header's own BlockId; entries are None for non-header blocks.
    pub loop_parent_header: Vec<Option<BlockId>>,
    /// Compile-time-known scalar seed values `(VarId, value)`, applied at VM
    /// init. Baked into the `.swcp` package so a concrete run is self-contained
    /// (no Python-AST-derived `native_meta.static_scalars` needed). Empty for the
    /// in-memory `lower()` path, which still takes static scalars via native_meta.
    pub static_scalars: Vec<(VarId, i64)>,
    /// Integer semantics the program was compiled under (see `SemanticsMode`).
    pub mode: SemanticsMode,
}

impl CompiledProgram {
    /// Rebuild lookup state needed by concrete execution after package
    /// deserialization. Terminators already contain `BlockId`s, so rebuilding
    /// `label_to_block` would duplicate every (often deeply inlined) block name
    /// without serving the VM. Keeping that derived map empty materially lowers
    /// peak memory for large packages while preserving block names themselves
    /// for diagnostics and coverage results.
    pub fn rebuild_runtime_name_index(&mut self) {
        self.label_to_block.clear();
        self.name_index = NameIndex::from_names(&self.var_names);
    }

    pub fn lookup_var(&self, name: &str) -> Option<VarId> {
        self.name_index.lookup(&self.var_names, name)
    }
}
