/// Variable ID — index into the VM's variable store.
pub type VarId = u32;
/// Block ID — index into the VM's block array.
pub type BlockId = u32;

/// A compiled expression — tree of Rust enums, no Python objects.
#[derive(Debug, Clone)]
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
    Builtin {
        fn_id: BuiltinFn,
        args: Vec<Expr>,
    },
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
}

/// Binary operators at the Boogie level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

/// A compiled statement.
#[derive(Debug, Clone)]
pub enum Stmt {
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
    /// havoc x, y, ...
    Havoc { vars: Vec<VarId> },
    /// havoc $CurrAddr (or .shadow) — allocation: read size from alloc_size_var,
    /// compute new_addr = (old_addr + size + 255) & ~255, set, trace, then clear.
    HavocCurrAddr { var_id: VarId, alloc_size_var: VarId },
    /// goto label
    Goto { targets: Vec<BlockId> },
    /// return
    Return,
    /// Calls that are ignored (printf, verifier_nondet, etc.)
    CallIgnored,
    /// call time.cross_product
    CallTime { assignments: Vec<VarId>, args: Vec<Expr> },
    /// call write.cross_product
    CallWrite { assignments: Vec<VarId>, args: Vec<Expr> },
    /// call read.cross_product
    CallRead { args: Vec<Expr> },
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct MemMapInfo {
    pub name: String,
    pub var_id: VarId,
    pub index_bit_width: u8,
    pub element_bit_width: u8,
}

/// A compiled program ready for VM execution.
#[derive(Debug)]
pub struct CompiledProgram {
    /// All blocks, indexed by BlockId
    pub blocks: Vec<Block>,
    /// Block name → BlockId
    pub label_to_block: rustc_hash::FxHashMap<String, BlockId>,
    /// Variable name → VarId
    pub var_names: Vec<String>,
    /// VarId → is_shadow
    pub is_shadow: Vec<bool>,
    /// Entry block ID
    pub entry_block: BlockId,
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
}
