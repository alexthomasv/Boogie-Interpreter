use crate::opcodes::*;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rustc_hash::FxHashMap;

/// Intern table: variable name → VarId
pub struct InternTable {
    map: FxHashMap<String, VarId>,
    names: Vec<String>,
    shadows: Vec<bool>,
}

impl InternTable {
    pub fn new() -> Self {
        Self {
            map: FxHashMap::default(),
            names: Vec::new(),
            shadows: Vec::new(),
        }
    }

    pub fn intern(&mut self, name: &str) -> VarId {
        if let Some(&id) = self.map.get(name) {
            return id;
        }
        let id = self.names.len() as VarId;
        self.names.push(name.to_string());
        self.shadows.push(name.ends_with(".shadow"));
        self.map.insert(name.to_string(), id);
        id
    }

    pub fn get(&self, name: &str) -> Option<VarId> {
        self.map.get(name).copied()
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn shadows(&self) -> &[bool] {
        &self.shadows
    }

    pub fn len(&self) -> u32 {
        self.names.len() as u32
    }
}

/// Lower a Python AST program into a CompiledProgram.
pub fn lower_program(py: Python<'_>, program: &Bound<'_, PyAny>) -> PyResult<CompiledProgram> {
    let mut intern = InternTable::new();
    let mut blocks = Vec::new();
    let mut label_to_block: FxHashMap<String, BlockId> = FxHashMap::default();
    let mut mem_maps = Vec::new();

    // Find the implementation declaration with entrypoint
    let declarations = program.getattr("declarations")?;
    let decls: &Bound<'_, PyList> = declarations.downcast()?;

    let mut impl_decl: Option<Bound<'_, PyAny>> = None;
    let mut proc_decl: Option<Bound<'_, PyAny>> = None;

    for decl in decls.iter() {
        let type_name = decl.get_type().name()?.to_string();
        if type_name == "ImplementationDeclaration" {
            let body = decl.getattr("body")?;
            if !body.is_none() {
                impl_decl = Some(decl.clone());
            }
        } else if type_name == "ProcedureDeclaration" {
            proc_decl = Some(decl.clone());
        }
    }

    let impl_d = impl_decl.expect("No ImplementationDeclaration with body found");

    // Process global declarations (initialize variables)
    for decl in decls.iter() {
        let type_name = decl.get_type().name()?.to_string();
        match type_name.as_str() {
            "StorageDeclaration" => {
                let names = decl.getattr("names")?;
                let names_list: &Bound<'_, PyList> = names.downcast()?;
                let type_obj = decl.getattr("type")?;
                let type_name_inner = type_obj.get_type().name()?.to_string();

                if type_name_inner == "MapType" {
                    assert!(names_list.len() == 1);
                    let name: String = names_list.get_item(0)?.extract()?;
                    let var_id = intern.intern(&name);
                    // Get bit widths
                    let bw = convert_type_to_bitwidth(py, &type_obj)?;
                    mem_maps.push(MemMapInfo {
                        name: name.clone(),
                        var_id,
                        index_bit_width: bw.0,
                        element_bit_width: bw.1,
                    });
                } else {
                    for item in names_list.iter() {
                        let name: String = item.extract()?;
                        intern.intern(&name);
                    }
                }
            }
            "ConstantDeclaration" => {
                let names = decl.getattr("names")?;
                let names_list: &Bound<'_, PyList> = names.downcast()?;
                for item in names_list.iter() {
                    let name: String = item.extract()?;
                    intern.intern(&name);
                }
            }
            _ => {}
        }
    }

    // Process local declarations from implementation body
    let body = impl_d.getattr("body")?;
    let locals = body.getattr("locals")?;
    let locals_list: &Bound<'_, PyList> = locals.downcast()?;
    for local_decl in locals_list.iter() {
        let type_name = local_decl.get_type().name()?.to_string();
        if type_name == "StorageDeclaration" {
            let names = local_decl.getattr("names")?;
            let names_list: &Bound<'_, PyList> = names.downcast()?;
            let type_obj = local_decl.getattr("type")?;
            let type_name_inner = type_obj.get_type().name()?.to_string();

            if type_name_inner == "MapType" {
                assert!(names_list.len() == 1);
                let name: String = names_list.get_item(0)?.extract()?;
                let var_id = intern.intern(&name);
                let bw = convert_type_to_bitwidth(py, &type_obj)?;
                mem_maps.push(MemMapInfo {
                    name: name.clone(),
                    var_id,
                    index_bit_width: bw.0,
                    element_bit_width: bw.1,
                });
            } else {
                for item in names_list.iter() {
                    let name: String = item.extract()?;
                    intern.intern(&name);
                }
            }
        }
    }

    // Process procedure parameters
    if let Some(ref pd) = proc_decl {
        let params = pd.getattr("parameters")?;
        let params_list: &Bound<'_, PyList> = params.downcast()?;
        for param in params_list.iter() {
            let names = param.getattr("names")?;
            let names_list: &Bound<'_, PyList> = names.downcast()?;
            for item in names_list.iter() {
                let name: String = item.extract()?;
                intern.intern(&name);
            }
        }
    }

    // First pass: build label → BlockId mapping
    let body_blocks = body.getattr("blocks")?;
    let body_blocks_list: &Bound<'_, PyList> = body_blocks.downcast()?;
    for (i, block) in body_blocks_list.iter().enumerate() {
        let name: String = block.getattr("name")?.extract()?;
        label_to_block.insert(name, i as BlockId);
    }

    // Second pass: lower blocks
    let mut pc: u32 = 0;
    for block_obj in body_blocks_list.iter() {
        let name: String = block_obj.getattr("name")?.extract()?;
        let block_id = label_to_block[&name];
        let start_pc = pc;

        let stmts = block_obj.getattr("statements")?;
        let stmts_list: &Bound<'_, PyList> = stmts.downcast()?;
        let n = stmts_list.len();

        let mut body_stmts = Vec::with_capacity(n.saturating_sub(1));
        let mut terminator = Stmt::Return;
        let mut assume_cond = None;

        for (i, stmt) in stmts_list.iter().enumerate() {
            let lowered = lower_stmt(py, &stmt, &mut intern, &label_to_block)?;
            if i < n - 1 {
                body_stmts.push(lowered);
                pc += 1;
            } else {
                terminator = lowered;
                // Terminator also occupies a PC slot (matching Python's initialize_code_metadata)
                pc += 1;
            }
        }

        // Extract assume condition (first non-trivial assume)
        // Look at the first body statements for the branch condition
        for stmt in &body_stmts {
            match stmt {
                Stmt::Assert { .. } => continue, // skip hhoudini asserts
                Stmt::Assume { expr } => {
                    assume_cond = Some(expr.clone());
                    break;
                }
                Stmt::AssumeTrue => continue,
                _ => break,
            }
        }

        blocks.push(Block {
            name,
            id: block_id,
            body: body_stmts,
            terminator,
            start_pc,
            assume_cond,
        });
    }

    // Post-pass: resolve $CurrAddr alloc_size_var by scanning Python AST
    // For each HavocCurrAddr, look at the block's statements at offset+3 and offset+5
    // to find the `$n` variable.
    for block_obj in body_blocks_list.iter() {
        let name: String = block_obj.getattr("name")?.extract()?;
        let block_id = label_to_block[&name];
        let stmts = block_obj.getattr("statements")?;
        let stmts_list: &Bound<'_, PyList> = stmts.downcast()?;

        // Find HavocCurrAddr positions in our compiled block
        let block = &blocks[block_id as usize];
        let mut patches: Vec<(usize, VarId)> = Vec::new();

        for (i, stmt) in block.body.iter().enumerate() {
            if let Stmt::HavocCurrAddr { alloc_size_var, .. } = stmt {
                if *alloc_size_var == u32::MAX {
                    // Find the alloc size var from Python statements at offsets +3 and +5
                    let utils = py.import_bound("utils.utils")?;
                    let mut size_var_id: Option<VarId> = None;

                    for offset in [3, 5] {
                        let idx = i + offset;
                        if idx < stmts_list.len() {
                            let py_stmt = stmts_list.get_item(idx)?;
                            let stmt_type = py_stmt.get_type().name()?.to_string();
                            if stmt_type == "AssumeStatement" {
                                let expr = py_stmt.getattr("expression")?;
                                let vars_set = utils.call_method1("extract_boogie_variables", (&expr,))?;
                                let vars: Vec<Bound<'_, PyAny>> = vars_set.iter()?.collect::<PyResult<Vec<_>>>()?;
                                for v in &vars {
                                    let vname: String = v.getattr("name")?.extract::<String>()?;
                                    if vname.ends_with("$n") || vname.ends_with("$n.shadow") {
                                        size_var_id = Some(intern.intern(&vname));
                                        break;
                                    }
                                }
                            }
                        }
                        if size_var_id.is_some() {
                            break;
                        }
                    }

                    if let Some(svid) = size_var_id {
                        patches.push((i, svid));
                    }
                }
            }
        }

        // Apply patches
        let block_mut = &mut blocks[block_id as usize];
        for (i, svid) in patches {
            if let Stmt::HavocCurrAddr { alloc_size_var, .. } = &mut block_mut.body[i] {
                *alloc_size_var = svid;
            }
        }
    }

    let curr_addr_id = intern.get("$CurrAddr");
    let curr_addr_shadow_id = intern.get("$CurrAddr.shadow");
    let m0_id = intern.get("$M.0");
    let m0_shadow_id = intern.get("$M.0.shadow");

    Ok(CompiledProgram {
        blocks,
        label_to_block,
        var_names: intern.names().to_vec(),
        is_shadow: intern.shadows().to_vec(),
        entry_block: 0,
        mem_maps,
        num_vars: intern.len(),
        curr_addr_id,
        curr_addr_shadow_id,
        m0_id,
        m0_shadow_id,
    })
}

/// Lower a Python statement to a Rust Stmt.
fn lower_stmt(
    py: Python<'_>,
    stmt: &Bound<'_, PyAny>,
    intern: &mut InternTable,
    label_map: &FxHashMap<String, BlockId>,
) -> PyResult<Stmt> {
    let type_name = stmt.get_type().name()?.to_string();

    match type_name.as_str() {
        "AssignStatement" => {
            let lhs_list = stmt.getattr("lhs")?;
            let rhs_list = stmt.getattr("rhs")?;
            let lhs_py: &Bound<'_, PyList> = lhs_list.downcast()?;
            let rhs_py: &Bound<'_, PyList> = rhs_list.downcast()?;

            if lhs_py.len() == 1 {
                let lhs_name: String = lhs_py.get_item(0)?.getattr("name")?.extract()?;
                let lhs_id = intern.intern(&lhs_name);
                let rhs_expr = lower_expr(py, &rhs_py.get_item(0)?, intern)?;
                Ok(Stmt::Assign1 {
                    lhs: lhs_id,
                    rhs: rhs_expr,
                })
            } else {
                let lhs: Vec<VarId> = lhs_py
                    .iter()
                    .map(|item| {
                        let name: String = item.getattr("name").unwrap().extract().unwrap();
                        intern.intern(&name)
                    })
                    .collect();
                let rhs: Vec<Expr> = rhs_py
                    .iter()
                    .map(|item| lower_expr(py, &item, intern).unwrap())
                    .collect();
                Ok(Stmt::AssignN { lhs, rhs })
            }
        }
        "AssertStatement" => {
            let expr = stmt.getattr("expression")?;
            let lowered = lower_expr(py, &expr, intern)?;
            Ok(Stmt::Assert { expr: lowered })
        }
        "AssumeStatement" => {
            let expr = stmt.getattr("expression")?;
            let expr_type = expr.get_type().name()?.to_string();

            // Check for assume true
            if expr_type == "BooleanLiteral" {
                let val: bool = expr.getattr("value")?.extract()?;
                if val {
                    return Ok(Stmt::AssumeTrue);
                }
            }

            // Check for quantified expression (memset/memcpy)
            if expr_type == "QuantifiedExpression" {
                return lower_quantified_assume(py, &expr, intern);
            }

            let lowered = lower_expr(py, &expr, intern)?;
            Ok(Stmt::Assume { expr: lowered })
        }
        "HavocStatement" => {
            let idents = stmt.getattr("identifiers")?;
            let idents_list: &Bound<'_, PyList> = idents.downcast()?;
            let mut vars = Vec::new();
            let mut curr_addr_var: Option<(String, VarId)> = None;
            for ident in idents_list.iter() {
                let name: String = ident.getattr("name")?.extract()?;
                let var_id = intern.intern(&name);
                if name == "$CurrAddr" || name == "$CurrAddr.shadow" {
                    curr_addr_var = Some((name, var_id));
                }
                vars.push(var_id);
            }
            if curr_addr_var.is_some() {
                // $CurrAddr havoc — store as placeholder; alloc_size_var filled in post-pass
                Ok(Stmt::HavocCurrAddr {
                    var_id: curr_addr_var.unwrap().1,
                    alloc_size_var: u32::MAX, // sentinel, filled in by post-pass
                })
            } else {
                Ok(Stmt::Havoc { vars })
            }
        }
        "GotoStatement" => {
            let idents = stmt.getattr("identifiers")?;
            let idents_list: &Bound<'_, PyList> = idents.downcast()?;
            let targets: Vec<BlockId> = idents_list
                .iter()
                .map(|ident| {
                    let name: String = ident.getattr("name").unwrap().extract().unwrap();
                    *label_map
                        .get(&name)
                        .unwrap_or_else(|| panic!("Unknown label: {}", name))
                })
                .collect();
            Ok(Stmt::Goto { targets })
        }
        "ReturnStatement" => Ok(Stmt::Return),
        "CallStatement" => lower_call(py, stmt, intern),
        _ => panic!("Unknown statement type: {}", type_name),
    }
}

/// Lower a call statement.
fn lower_call(
    py: Python<'_>,
    stmt: &Bound<'_, PyAny>,
    intern: &mut InternTable,
) -> PyResult<Stmt> {
    let proc = stmt.getattr("procedure")?;
    let proc_name: String = proc.getattr("name")?.extract()?;

    // Check ignore patterns (simplified — match the Python regex patterns)
    if is_ignored_call(&proc_name) {
        return Ok(Stmt::CallIgnored);
    }

    if proc_name == "putc.cross_product" {
        return Ok(Stmt::CallIgnored);
    }

    if proc_name == "time.cross_product" {
        let args = lower_call_args(py, stmt, intern)?;
        let assignments = lower_call_assignments(stmt, intern)?;
        return Ok(Stmt::CallTime { assignments, args });
    }

    if proc_name == "write.cross_product" {
        let args = lower_call_args(py, stmt, intern)?;
        let assignments = lower_call_assignments(stmt, intern)?;
        return Ok(Stmt::CallWrite { assignments, args });
    }

    if proc_name == "read.cross_product" {
        let args = lower_call_args(py, stmt, intern)?;
        return Ok(Stmt::CallRead { args });
    }

    panic!("Unknown call: {}", proc_name);
}

fn lower_call_args(
    py: Python<'_>,
    stmt: &Bound<'_, PyAny>,
    intern: &mut InternTable,
) -> PyResult<Vec<Expr>> {
    let args = stmt.getattr("arguments")?;
    let args_list: &Bound<'_, PyList> = args.downcast()?;
    let mut result = Vec::new();
    for arg in args_list.iter() {
        result.push(lower_expr(py, &arg, intern)?);
    }
    Ok(result)
}

fn lower_call_assignments(
    stmt: &Bound<'_, PyAny>,
    intern: &mut InternTable,
) -> PyResult<Vec<VarId>> {
    let assigns = stmt.getattr("assignments")?;
    let assigns_list: &Bound<'_, PyList> = assigns.downcast()?;
    let mut result = Vec::new();
    for a in assigns_list.iter() {
        let name: String = a.getattr("name")?.extract()?;
        result.push(intern.intern(&name));
    }
    Ok(result)
}

/// Check if a call should be ignored (matches Python's CALL_IGNORE_FN_PATTERNS).
fn is_ignored_call(name: &str) -> bool {
    // __VERIFIER_nondet_*
    if name.starts_with("__VERIFIER_nondet_") {
        return true;
    }
    // boogie_si_record_*
    if name.starts_with("boogie_si_record_") {
        return true;
    }
    // $initialize.cross_product
    if name == "$initialize.cross_product" {
        return true;
    }
    // $alloc
    if name == "$alloc" {
        return true;
    }
    // corral_atomic_begin/end
    if name.starts_with("corral_atomic_") {
        return true;
    }
    // printf.*
    if name.starts_with("printf.ref") || name == "printf" {
        return true;
    }
    // __SMACK_value*
    if name.starts_with("__SMACK_value") || name.starts_with("__SMACK_values") {
        return true;
    }
    // llvm.lifetime.*
    if name.contains("llvm.lifetime.") {
        return true;
    }
    false
}

/// Lower a quantified assume (memset/memcpy patterns).
fn lower_quantified_assume(
    py: Python<'_>,
    q_expr: &Bound<'_, PyAny>,
    intern: &mut InternTable,
) -> PyResult<Stmt> {
    let expression = q_expr.getattr("expression")?;
    let op: String = expression.getattr("op")?.extract()?;

    // Get all boogie variables in the expression
    let utils = py.import_bound("utils.utils")?;
    let boogie_vars_set = utils.call_method1("extract_boogie_variables", (&expression,))?;
    let boogie_vars: Vec<Bound<'_, PyAny>> = boogie_vars_set.iter()?.collect::<PyResult<Vec<_>>>()?;

    // Classify by checking variable name suffixes
    let mut var_names: Vec<(String, VarId)> = Vec::new();
    for v in &boogie_vars {
        let name: String = v.getattr("name")?.extract::<String>()?;
        let var_id = intern.intern(&name);
        var_names.push((name, var_id));
    }

    let is_memset = var_names.iter().any(|(n, _)| n.contains("memset"));
    let is_memcpy = var_names.iter().any(|(n, _)| n.contains("memcpy"));

    if is_memset {
        lower_memset_assume(py, &expression, &op, &var_names, intern, q_expr)
    } else if is_memcpy {
        lower_memcpy_assume(py, &expression, &op, &var_names, intern, q_expr)
    } else {
        panic!("Unknown quantified assume pattern");
    }
}

fn lower_memset_assume(
    _py: Python<'_>,
    expression: &Bound<'_, PyAny>,
    op: &str,
    var_names: &[(String, VarId)],
    intern: &mut InternTable,
    q_expr: &Bound<'_, PyAny>,
) -> PyResult<Stmt> {
    // Get free variables (not in quantifier variable list)
    let q_variables = q_expr.getattr("variables")?;
    let q_vars_list: &Bound<'_, PyList> = q_variables.downcast()?;
    let q_var_names: Vec<String> = q_vars_list
        .iter()
        .map(|v| {
            // Quantifier variables are StorageDeclaration objects with `names` list
            let names = v.getattr("names").unwrap();
            let names_list = names.downcast::<PyList>().unwrap();
            names_list.get_item(0).unwrap().extract::<String>().unwrap()
        })
        .collect();

    let free_vars: Vec<&(String, VarId)> = var_names
        .iter()
        .filter(|(n, _)| !q_var_names.contains(n))
        .collect();

    match op {
        "&&" => {
            let m_ret = find_var_by_suffix_ref(&free_vars, "M.ret");
            let dst = find_var_by_suffix_ref(&free_vars, "dst");
            let len = find_var_by_suffix_ref(&free_vars, "len");
            let val = find_var_by_suffix_ref(&free_vars, "val");
            Ok(Stmt::QuantMemsetWrite {
                m_ret,
                dst,
                len,
                val,
            })
        }
        "==>" => {
            let lhs = expression.getattr("lhs")?;
            let fn_obj = lhs.getattr("function")?;
            let fn_name: String = fn_obj.getattr("name")?.extract()?;

            if fn_name.starts_with("$slt") {
                let m_ret = find_var_by_suffix_ref(&free_vars, "M.ret");
                let m_src = find_var_by_suffix_either_ref(&free_vars, &["M"]);
                let dst = find_var_by_suffix_ref(&free_vars, "dst");
                Ok(Stmt::QuantMemsetPreserveLt { m_ret, m_src, dst })
            } else if fn_name.starts_with("$sle") {
                let m_ret = find_var_by_suffix_ref(&free_vars, "M.ret");
                let m_src = find_var_by_suffix_either_ref(&free_vars, &["M"]);
                let dst = find_var_by_suffix_ref(&free_vars, "dst");
                let len = find_var_by_suffix_ref(&free_vars, "len");
                Ok(Stmt::QuantMemsetPreserveGe {
                    m_ret,
                    m_src,
                    dst,
                    len,
                })
            } else {
                panic!("Unknown memset quantified expression fn: {}", fn_name);
            }
        }
        _ => panic!("Unknown memset quantified expression op: {}", op),
    }
}

fn lower_memcpy_assume(
    _py: Python<'_>,
    expression: &Bound<'_, PyAny>,
    op: &str,
    var_names: &[(String, VarId)],
    intern: &mut InternTable,
    _q_expr: &Bound<'_, PyAny>,
) -> PyResult<Stmt> {
    match op {
        "&&" => {
            let m_ret = find_var_by_suffix_slice(var_names, "$M.ret");
            let m_src = find_var_by_suffix_slice(var_names, "$M.src");
            let dst = find_var_by_suffix_slice(var_names, "$dst");
            let src = find_var_by_suffix_slice(var_names, "$src");
            let len = find_var_by_suffix_slice(var_names, "$len");
            Ok(Stmt::QuantMemcpyWrite {
                m_ret,
                m_src,
                dst,
                src,
                len,
            })
        }
        "==>" => {
            let lhs = expression.getattr("lhs")?;
            let fn_obj = lhs.getattr("function")?;
            let fn_name: String = fn_obj.getattr("name")?.extract()?;

            if fn_name.starts_with("$slt") {
                let m_ret = find_var_by_suffix_slice(var_names, "$M.ret");
                let m_src = find_var_by_suffix_slice(var_names, "$M.dst");
                let dst = find_var_by_suffix_slice(var_names, "$dst");
                Ok(Stmt::QuantMemcpyPreserveLt { m_ret, m_src, dst })
            } else if fn_name.starts_with("$sle") {
                let m_ret = find_var_by_suffix_slice(var_names, "$M.ret");
                let m_src = find_var_by_suffix_slice(var_names, "$M.dst");
                let dst = find_var_by_suffix_slice(var_names, "$dst");
                let len = find_var_by_suffix_slice(var_names, "$len");
                Ok(Stmt::QuantMemcpyPreserveGe {
                    m_ret,
                    m_src,
                    dst,
                    len,
                })
            } else {
                panic!("Unknown memcpy quantified expression fn: {}", fn_name);
            }
        }
        _ => panic!("Unknown memcpy quantified expression op: {}", op),
    }
}

/// Find a variable by suffix match in a slice of (name, var_id) tuples.
fn find_var_by_suffix_slice(vars: &[(String, VarId)], suffix: &str) -> VarId {
    for (name, var_id) in vars {
        if name.ends_with(suffix) || name.ends_with(&format!("{}.shadow", suffix)) {
            return *var_id;
        }
    }
    panic!("No variable found with suffix: {}", suffix);
}

/// Find a variable by suffix match in a Vec of references.
fn find_var_by_suffix_ref(vars: &[&(String, VarId)], suffix: &str) -> VarId {
    for (name, var_id) in vars {
        if name.ends_with(suffix) || name.ends_with(&format!("{}.shadow", suffix)) {
            return *var_id;
        }
    }
    panic!("No variable found with suffix: {}", suffix);
}

fn find_var_by_suffix_either_ref(vars: &[&(String, VarId)], suffixes: &[&str]) -> VarId {
    for (name, var_id) in vars {
        for suffix in suffixes {
            if name.ends_with(suffix) || name.ends_with(&format!("{}.shadow", suffix)) {
                return *var_id;
            }
        }
    }
    panic!("No variable found with suffixes: {:?}", suffixes);
}

/// Lower a Python expression to a Rust Expr.
fn lower_expr(
    py: Python<'_>,
    expr: &Bound<'_, PyAny>,
    intern: &mut InternTable,
) -> PyResult<Expr> {
    let type_name = expr.get_type().name()?.to_string();

    match type_name.as_str() {
        "StorageIdentifier" | "ProcedureIdentifier" => {
            let name: String = expr.getattr("name")?.extract()?;
            let var_id = intern.intern(&name);
            Ok(Expr::Var(var_id))
        }
        "IntegerLiteral" => {
            // Boogie integers can be arbitrarily large; truncate to i64
            let value_obj = expr.getattr("value")?;
            let value: i64 = match value_obj.extract::<i64>() {
                Ok(v) => v,
                Err(_) => {
                    // Large integer — extract as u64 or truncate
                    match value_obj.extract::<u64>() {
                        Ok(v) => v as i64,
                        Err(_) => {
                            // Very large — mask to 64 bits
                            let py_int = value_obj.call_method1("__and__", (u64::MAX,))?;
                            py_int.extract::<u64>()? as i64
                        }
                    }
                }
            };
            Ok(Expr::Const(value))
        }
        "BooleanLiteral" => {
            let value: bool = expr.getattr("value")?.extract()?;
            Ok(Expr::Bool(value))
        }
        "FunctionApplication" => {
            let func = expr.getattr("function")?;
            let f_name: String = func.getattr("name")?.extract()?;
            let args = expr.getattr("arguments")?;
            let args_list: &Bound<'_, PyList> = args.downcast()?;

            // Store functions
            if matches!(
                f_name.as_str(),
                "$store.i8" | "$store.i16" | "$store.i32" | "$store.i64" | "$store.ref"
            ) {
                let bw = store_load_bitwidth(&f_name);
                let map = lower_expr(py, &args_list.get_item(0)?, intern)?;
                let index = lower_expr(py, &args_list.get_item(1)?, intern)?;
                let value = lower_expr(py, &args_list.get_item(2)?, intern)?;
                return Ok(Expr::Store {
                    bit_width: bw,
                    map: Box::new(map),
                    index: Box::new(index),
                    value: Box::new(value),
                });
            }

            // Load functions
            if matches!(
                f_name.as_str(),
                "$load.i8" | "$load.i16" | "$load.i32" | "$load.i64" | "$load.ref"
            ) {
                let bw = store_load_bitwidth(&f_name);
                let map = lower_expr(py, &args_list.get_item(0)?, intern)?;
                let index = lower_expr(py, &args_list.get_item(1)?, intern)?;
                return Ok(Expr::Load {
                    bit_width: bw,
                    map: Box::new(map),
                    index: Box::new(index),
                });
            }

            // $isExternal
            if f_name == "$isExternal" {
                return Ok(Expr::IsExternal);
            }

            // Resolve builtin function
            if let Some(fn_id) = resolve_builtin(&f_name) {
                let lowered_args: Vec<Expr> = args_list
                    .iter()
                    .map(|a| lower_expr(py, &a, intern).unwrap())
                    .collect();
                return Ok(Expr::Builtin {
                    fn_id,
                    args: lowered_args,
                });
            }

            panic!("Unknown function: {}", f_name);
        }
        "BinaryExpression" => {
            let op_str: String = expr.getattr("op")?.extract()?;
            let lhs = expr.getattr("lhs")?;
            let rhs = expr.getattr("rhs")?;
            let op = match op_str.as_str() {
                "==" => BinOp::Eq,
                "!=" => BinOp::Ne,
                "<" => BinOp::Lt,
                ">" => BinOp::Gt,
                ">=" => BinOp::Ge,
                "<=" => BinOp::Le,
                "&&" => BinOp::And,
                "||" => BinOp::Or,
                "==>" => BinOp::Implies,
                "<==>" => BinOp::Iff,
                "-" => BinOp::Sub,
                "*" => BinOp::Mul,
                "+" => BinOp::Add,
                _ => panic!("Unknown binary op: {}", op_str),
            };
            let lhs_expr = lower_expr(py, &lhs, intern)?;
            let rhs_expr = lower_expr(py, &rhs, intern)?;
            Ok(Expr::BinOp {
                op,
                lhs: Box::new(lhs_expr),
                rhs: Box::new(rhs_expr),
            })
        }
        "LogicalNegation" => {
            let inner = expr.getattr("expression")?;
            let lowered = lower_expr(py, &inner, intern)?;
            Ok(Expr::Not(Box::new(lowered)))
        }
        "IfExpression" => {
            let cond = expr.getattr("condition")?;
            let then = expr.getattr("then")?;
            let else_ = expr.getattr("else_")?;
            let cond_expr = lower_expr(py, &cond, intern)?;
            let then_expr = lower_expr(py, &then, intern)?;
            let else_expr = lower_expr(py, &else_, intern)?;
            Ok(Expr::IfThenElse {
                cond: Box::new(cond_expr),
                then_: Box::new(then_expr),
                else_: Box::new(else_expr),
            })
        }
        "UnaryExpression" => {
            // Unary minus — just evaluate the inner expression (same as Python)
            let inner = expr.getattr("expression")?;
            lower_expr(py, &inner, intern)
        }
        _ => panic!("Unknown expression type: {}", type_name),
    }
}

/// Get bit width from store/load function name.
fn store_load_bitwidth(name: &str) -> u8 {
    match name.rsplit('.').next().unwrap() {
        "i8" => 8,
        "i16" => 16,
        "i32" => 32,
        "i64" | "ref" => 64,
        s => panic!("Unknown store/load type: {}", s),
    }
}

/// Resolve a function name to a BuiltinFn.
fn resolve_builtin(name: &str) -> Option<BuiltinFn> {
    // This maps all entries from fn_map_to_op and generate_function_map
    let result = match name {
        // Mul
        "$mul.ref" | "$mul.i64" => BuiltinFn::Mul { bits: 64 },
        "$mul.i32" => BuiltinFn::Mul { bits: 32 },
        "$mul.i8" => BuiltinFn::Mul { bits: 8 },

        // Add
        "$add.ref" | "$add.i64" => BuiltinFn::Add { bits: 64 },
        "$add.i32" => BuiltinFn::Add { bits: 32 },
        "$add.i8" => BuiltinFn::Add { bits: 8 },

        // Sub
        "$sub.ref" | "$sub.i64" => BuiltinFn::Sub { bits: 64 },
        "$sub.i32" => BuiltinFn::Sub { bits: 32 },
        "$sub.i16" => BuiltinFn::Sub { bits: 16 },
        "$sub.i8" => BuiltinFn::Sub { bits: 8 },

        // Not (unary)
        "$not.i1" => BuiltinFn::Not { bits: 1 },
        "$not.i8" => BuiltinFn::Not { bits: 8 },
        "$not.i16" => BuiltinFn::Not { bits: 16 },
        "$not.i32" => BuiltinFn::Not { bits: 32 },
        "$not.i64" | "$not.ref" => BuiltinFn::Not { bits: 64 },

        // And
        "$and.ref" | "$and.i64" => BuiltinFn::And { bits: 64 },
        "$and.i32" => BuiltinFn::And { bits: 32 },
        "$and.i8" => BuiltinFn::And { bits: 8 },
        "$and.i1" => BuiltinFn::And { bits: 1 },

        // Or
        "$or.ref" | "$or.i64" => BuiltinFn::Or { bits: 64 },
        "$or.i32" => BuiltinFn::Or { bits: 32 },
        "$or.i8" => BuiltinFn::Or { bits: 8 },
        "$or.i1" => BuiltinFn::Or { bits: 1 },

        // Xor
        "$xor.ref" | "$xor.i64" => BuiltinFn::Xor { bits: 64 },
        "$xor.i32" => BuiltinFn::Xor { bits: 32 },
        "$xor.i8" => BuiltinFn::Xor { bits: 8 },
        "$xor.i1" => BuiltinFn::Xor { bits: 1 },

        // Equality
        "$ne.ref" | "$ne.i64" => BuiltinFn::BvNe { bits: 64 },
        "$ne.i32" => BuiltinFn::BvNe { bits: 32 },
        "$ne.i8" => BuiltinFn::BvNe { bits: 8 },

        "$eq.ref" | "$eq.i64" => BuiltinFn::BvEq { bits: 64 },
        "$eq.i32" => BuiltinFn::BvEq { bits: 32 },
        "$eq.i8" => BuiltinFn::BvEq { bits: 8 },
        "$eq.i1" => BuiltinFn::BvEq { bits: 1 },

        // Unsigned div/cmp
        "$udiv.ref" | "$udiv.i64" => BuiltinFn::Udiv { bits: 64 },
        "$udiv.i32" => BuiltinFn::Udiv { bits: 32 },
        "$udiv.i8" => BuiltinFn::Udiv { bits: 8 },

        "$sdiv.i64" => BuiltinFn::Sdiv { bits: 64 },
        "$sdiv.i32" => BuiltinFn::Sdiv { bits: 32 },

        "$ult.ref" | "$ult.i64" => BuiltinFn::Ult { bits: 64 },
        "$ult.i32" => BuiltinFn::Ult { bits: 32 },

        "$ugt.i64" => BuiltinFn::Ugt { bits: 64 },
        "$ugt.i32" => BuiltinFn::Ugt { bits: 32 },

        "$uge.i64" => BuiltinFn::Uge { bits: 64 },
        "$uge.i32" => BuiltinFn::Uge { bits: 32 },

        // Signed comparisons
        "$sgt.ref.bool" => BuiltinFn::SgtBool { bits: 64 },
        "$sgt.i32" => BuiltinFn::Sgt { bits: 32 },
        "$sgt.i64" => BuiltinFn::Sgt { bits: 64 },

        "$sge.ref.bool" => BuiltinFn::SgeBool { bits: 64 },
        "$sge.i32" => BuiltinFn::Sge { bits: 32 },
        "$sge.i64" => BuiltinFn::Sge { bits: 64 },

        "$sle.i32" => BuiltinFn::Sle { bits: 32 },
        "$sle.i64" => BuiltinFn::Sle { bits: 64 },
        "$sle.ref.bool" => BuiltinFn::SleBool { bits: 64 },

        "$slt.ref.bool" => BuiltinFn::SltBool { bits: 64 },
        "$slt.i64" => BuiltinFn::Slt { bits: 64 },
        "$slt.i32" => BuiltinFn::Slt { bits: 32 },
        "$slt.i8" => BuiltinFn::Slt { bits: 8 },

        "$ule.i64" => BuiltinFn::Ule { bits: 64 },
        "$ule.i32" => BuiltinFn::Ule { bits: 32 },
        "$ule.i8" => BuiltinFn::Ule { bits: 8 },

        // Remainder
        "$urem.i64" => BuiltinFn::Urem { bits: 64 },
        "$urem.i32" => BuiltinFn::Urem { bits: 32 },
        "$urem.i8" => BuiltinFn::Urem { bits: 8 },

        "$srem.i64" => BuiltinFn::Srem { bits: 64 },
        "$srem.i32" => BuiltinFn::Srem { bits: 32 },
        "$srem.i8" => BuiltinFn::Srem { bits: 8 },

        // Shifts
        "$shl.i64" => BuiltinFn::Shl { bits: 64 },
        "$shl.i32" => BuiltinFn::Shl { bits: 32 },

        "$lshr.i64" => BuiltinFn::Lshr { bits: 64 },
        "$lshr.i32" => BuiltinFn::Lshr { bits: 32 },

        "$ashr.i64" => BuiltinFn::Ashr { bits: 64 },
        "$ashr.i32" => BuiltinFn::Ashr { bits: 32 },

        // Casts
        "$bitcast.ref.ref" => BuiltinFn::Bitcast,
        "$p2i.ref.i64" => BuiltinFn::P2i,
        "$i2p.i64.ref" => BuiltinFn::I2p,

        // Sign extension
        "$sext.i32.i64" => BuiltinFn::Sext { src: 32, dst: 64 },
        "$sext.i8.i32" => BuiltinFn::Sext { src: 8, dst: 32 },
        "$sext.i16.i32" => BuiltinFn::Sext { src: 16, dst: 32 },

        // Zero extension
        "$zext.i32.i64" => BuiltinFn::Zext { src: 32, dst: 64 },
        "$zext.i8.i32" => BuiltinFn::Zext { src: 8, dst: 32 },
        "$zext.i8.i64" => BuiltinFn::Zext { src: 8, dst: 64 },
        "$zext.i1.i32" => BuiltinFn::Zext { src: 1, dst: 32 },
        "$zext.i1.i64" => BuiltinFn::Zext { src: 1, dst: 64 },
        "$zext.i16.i32" => BuiltinFn::Zext { src: 16, dst: 32 },
        "$zext.i16.i64" => BuiltinFn::Zext { src: 16, dst: 64 },

        // Truncation
        "$trunc.i32.i8" => BuiltinFn::Trunc { dst: 8 },
        "$trunc.i32.i16" => BuiltinFn::Trunc { dst: 16 },
        "$trunc.i64.i8" => BuiltinFn::Trunc { dst: 8 },
        "$trunc.i64.i16" => BuiltinFn::Trunc { dst: 16 },
        "$trunc.i64.i32" => BuiltinFn::Trunc { dst: 32 },
        "$trunc.i32.i1" => BuiltinFn::Trunc { dst: 1 },
        "$trunc.i64.i1" => BuiltinFn::Trunc { dst: 1 },

        _ => return None,
    };
    Some(result)
}

/// Convert a Python MapType to (index_bw, element_bw).
fn convert_type_to_bitwidth(py: Python<'_>, type_obj: &Bound<'_, PyAny>) -> PyResult<(u8, u8)> {
    let domain = type_obj.getattr("domain")?;
    let range = type_obj.getattr("range")?;
    let domain_list: &Bound<'_, PyList> = domain.downcast()?;
    assert!(domain_list.len() == 1, "Only single-index maps supported");
    let domain_bw = scalar_type_bitwidth(py, &domain_list.get_item(0)?)?;
    let range_bw = scalar_type_bitwidth(py, &range)?;
    Ok((domain_bw, range_bw))
}

/// Get bit width for a scalar type.
fn scalar_type_bitwidth(_py: Python<'_>, type_obj: &Bound<'_, PyAny>) -> PyResult<u8> {
    let type_name = type_obj.get_type().name()?.to_string();
    match type_name.as_str() {
        "BooleanType" => Ok(1),
        "IntegerType" => Ok(32),
        "CustomType" => {
            let name: String = type_obj.getattr("name")?.extract()?;
            match name.as_str() {
                "i1" | "bool" => Ok(1),
                "i8" => Ok(8),
                "i16" => Ok(16),
                "i32" => Ok(32),
                "i64" | "ref" | "$mop" => Ok(64),
                _ => panic!("Unknown custom type: {}", name),
            }
        }
        _ => panic!("Unknown type: {}", type_name),
    }
}
