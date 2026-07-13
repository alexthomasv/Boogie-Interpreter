/// Variable ID — index into the VM's variable store.
pub type VarId = u32;
/// Block ID — index into the VM's block array.
pub type BlockId = u32;

use serde::{Deserialize, Serialize};

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
    /// Memory store: $store.iN(map, index, value)
    Store {
        bit_width: u8,
        map: Box<Expr>,
        index: Box<Expr>,
        value: Box<Expr>,
    },
    /// Memory load: $load.iN(map, index)
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
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SemanticsMode {
    /// SMACK unbounded-integer encoding (`type i32 = int`, plain `(i1+i2)` prelude).
    Int,
    /// SMACK bit-vector encoding — wrapping algebra is the correct model.
    /// Default: every pre-mode artifact was built under BV assumptions.
    #[default]
    Bv,
}

impl SemanticsMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            SemanticsMode::Int => "int",
            SemanticsMode::Bv => "bv",
        }
    }

    pub fn from_str_opt(s: Option<&str>) -> Result<Self, String> {
        match s {
            None => Ok(SemanticsMode::Bv),
            Some(v) => match v {
                "int" => Ok(SemanticsMode::Int),
                "bv" => Ok(SemanticsMode::Bv),
                other => Err(format!(
                    "unknown semantics mode {:?} (expected \"int\" or \"bv\")",
                    other
                )),
            },
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
    pub var_names: Vec<String>,
    /// Variable name → VarId (derived; rebuilt on load via `rebuild_lookup_maps`, not serialized)
    #[serde(skip)]
    pub name_to_var: rustc_hash::FxHashMap<String, VarId>,
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
    #[serde(default)]
    pub static_scalars: Vec<(VarId, i64)>,
    /// Integer semantics the program was compiled under (see `SemanticsMode`).
    /// `#[serde(default)]` documents intent (Bv) for self-describing formats;
    /// note the bincode `.swcp` format is positional, so a pre-mode `.swcp`
    /// fails to deserialize LOUDLY rather than silently defaulting.
    #[serde(default)]
    pub mode: SemanticsMode,
}

impl CompiledProgram {
    /// Rebuild the derived lookup maps (`label_to_block`, `name_to_var`) after
    /// deserialization, where they are `#[serde(skip)]` and arrive empty. For
    /// duplicate block labels the *last* block wins, matching the live build in
    /// `lowering::lower_program_full` and Python's `initialize_code_metadata`.
    pub fn rebuild_lookup_maps(&mut self) {
        let mut label_to_block: rustc_hash::FxHashMap<String, BlockId> =
            rustc_hash::FxHashMap::default();
        for b in &self.blocks {
            label_to_block.insert(b.name.clone(), b.id);
        }
        self.label_to_block = label_to_block;

        let mut name_to_var: rustc_hash::FxHashMap<String, VarId> =
            rustc_hash::FxHashMap::default();
        for (i, n) in self.var_names.iter().enumerate() {
            name_to_var.insert(n.clone(), i as VarId);
        }
        self.name_to_var = name_to_var;
    }
}
