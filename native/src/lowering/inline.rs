//! Native inlining → compact bytecode.
//!
//! Inlines `{:inline}` procedures while lowering the (un-inlined) Python AST
//! straight into the VM's `CompiledProgram` IR — no Boogie `/printInlined`, no
//! multi-GB text round-trip. The result is differential-tested against Boogie's
//! inliner as the oracle (see `tools/test_inline_equiv.py`): a missed inline
//! would silently become `CallNondet` havoc, so correctness is non-negotiable.
//!
//! Variable renaming is handled entirely by the frame-aware `InternTable`
//! (`super::InternTable`): while a callee body is lowered under its `Frame`, the
//! callee's local/param/return names are prefixed `inline$<proc>$<N>$…` (N a
//! global instance counter) so each inline instance gets fresh, collision-free
//! variables, while globals stay shared. This module only wires up call
//! argument/return bindings, block splitting at call sites, and goto relinking.

use super::fold::{fold_in_place, fold_stmt};
use super::{
    convert_type_to_bitwidth, lower_call, lower_entry_preconditions, lower_expr, lower_stmt, Frame,
    InternTable,
};
use crate::opcodes::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustc_hash::{FxHashMap, FxHashSet};

/// A statement in a proto-block. Non-goto statements are lowered eagerly (var
/// interning done under the correct frame); gotos carry resolved target *names*
/// that become `BlockId`s in pass 2, once every block id is assigned.
enum ProtoStmt {
    Lowered(Stmt),
    Goto(Vec<String>),
}

struct ProtoBlock {
    name: String,
    /// Body + terminator; the terminator is the last element.
    stmts: Vec<ProtoStmt>,
}

struct Inliner<'py> {
    py: Python<'py>,
    intern: InternTable,
    mem_maps: Vec<MemMapInfo>,
    /// Output proto-blocks, in execution order. Index == future BlockId.
    proto: Vec<ProtoBlock>,
    /// The block currently being accumulated (None between blocks).
    cur: Option<ProtoBlock>,
    /// Impl name → its ImplementationDeclaration (has body/params/returns).
    body_by_name: FxHashMap<String, Bound<'py, PyAny>>,
    /// Names of procedures to inline ({:inline}, bodied, non-entry).
    inline_names: FxHashSet<String>,
    /// Global instance counter — unique per inline expansion.
    counter: u32,
    /// Procedures on the current DFS path (the recursion bound).
    active: FxHashSet<String>,
    /// Count of recursive call sites left as residual havoc (Boogie {:inline 1}).
    bounded_recursions: usize,
    /// Count of blocks whose tail was dropped at an `assume false` (peak lever).
    dead_block_cutoffs: usize,
    /// Empty label map for `lower_stmt` on plain statements (which never read
    /// it — gotos are handled here, and If/While bodies carry no gotos).
    empty_labels: FxHashMap<String, BlockId>,
}

/// Inline `{:inline}` procedures and lower to a `CompiledProgram`. The program
/// must be the *un-inlined* shadowed AST with the entry marked `{:entrypoint}`.
pub fn inline_lower_program<'py>(
    py: Python<'py>,
    program: &Bound<'py, PyAny>,
    static_scalars: Option<&Bound<'py, PyDict>>,
    mode: crate::opcodes::SemanticsMode,
) -> PyResult<CompiledProgram> {
    let declarations = program.getattr("declarations")?;
    let decls: &Bound<'_, PyList> = declarations.downcast()?;

    // Build body_by_name / inline_names and find the entry impl.
    let mut body_by_name: FxHashMap<String, Bound<'_, PyAny>> = FxHashMap::default();
    let mut inline_names: FxHashSet<String> = FxHashSet::default();
    let mut entrypoint_names: FxHashSet<String> = FxHashSet::default();

    let mut inline_marked: FxHashSet<String> = FxHashSet::default();
    for decl in decls.iter() {
        let tn = decl.get_type().name()?.to_string();
        let is_proc = tn == "ProcedureDeclaration" || tn == "ImplementationDeclaration";
        if !is_proc {
            continue;
        }
        let name: String = decl.getattr("name")?.extract()?;
        if decl
            .call_method1("has_attribute", ("entrypoint",))?
            .extract()?
        {
            entrypoint_names.insert(name.clone());
        }
        if decl.call_method1("has_attribute", ("inline",))?.extract()? {
            inline_marked.insert(name.clone());
        }
        // The body may live on a `procedure …{}` (combined form — the SMACK
        // input) or a separate `implementation …{}` (Boogie's split output).
        let body = decl.getattr("body")?;
        if !body.is_none() {
            body_by_name.insert(name, decl.clone());
        }
    }
    // Inline-set = marked AND bodied AND not the entry.
    for name in body_by_name.keys() {
        if inline_marked.contains(name) && !entrypoint_names.contains(name) {
            inline_names.insert(name.clone());
        }
    }

    // Locate the entry impl: prefer {:entrypoint}, else the lone bodied impl
    // that is not marked {:inline}.
    let entry: Bound<'_, PyAny> = {
        let mut found = None;
        for (name, decl) in &body_by_name {
            if entrypoint_names.contains(name) {
                found = Some(decl.clone());
                break;
            }
        }
        if found.is_none() {
            let non_inline: Vec<&Bound<'_, PyAny>> = body_by_name
                .iter()
                .filter(|(n, _)| !inline_names.contains(*n))
                .map(|(_, d)| d)
                .collect();
            if non_inline.len() == 1 {
                found = Some(non_inline[0].clone());
            }
        }
        found.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "inline_lower_program: could not identify the entry implementation \
                 (no {:entrypoint} impl and not exactly one non-inlined bodied impl)",
            )
        })?
    };
    let entry_name: String = entry.getattr("name")?.extract()?;

    let mut inl = Inliner {
        py,
        intern: InternTable::new_with_mode(mode),
        mem_maps: Vec::new(),
        proto: Vec::new(),
        cur: None,
        body_by_name,
        inline_names,
        counter: 0,
        active: FxHashSet::default(),
        bounded_recursions: 0,
        dead_block_cutoffs: 0,
        empty_labels: FxHashMap::default(),
    };

    if std::env::var("INLINE_DEBUG").is_ok() {
        eprintln!(
            "[inline_lower] entry={} | inline_names={} | body_by_name={} | entrypoints={}",
            entry_name,
            inl.inline_names.len(),
            inl.body_by_name.len(),
            entrypoint_names.len()
        );
    }

    // Intern globals (and global memory maps), then the entry's own locals and
    // params — mirroring the prelude of `lower_program_full`.
    inl.process_globals(decls)?;
    inl.process_impl_locals(&entry, /*frame*/ None)?;
    inl.process_impl_params(&entry, /*frame*/ None)?;

    let entry_preconditions = lower_entry_preconditions(py, &entry, &mut inl.intern)?;

    // Inline-lower the entry's blocks (no frame, returns stay `Return`).
    inl.active.insert(entry_name.clone());
    inl.inline_impl(&entry, None, None)?;
    inl.active.remove(&entry_name);

    if inl.bounded_recursions > 0 {
        eprintln!(
            "[inline_lower] bounded {} recursive call site(s) as residual havoc \
             (Boogie {{:inline 1}} semantics)",
            inl.bounded_recursions
        );
    }
    if inl.dead_block_cutoffs > 0 {
        eprintln!(
            "[inline_lower] cut {} block tail(s) at `assume false` (skipped \
             inlining unreachable code)",
            inl.dead_block_cutoffs
        );
    }

    // Assign BlockIds and build label_to_block (last wins on duplicate labels,
    // matching the live build + Python's initialize_code_metadata).
    let mut label_to_block: FxHashMap<String, BlockId> = FxHashMap::default();
    for (i, pb) in inl.proto.iter().enumerate() {
        label_to_block.insert(pb.name.clone(), i as BlockId);
    }

    // Pass 2: resolve gotos and assemble blocks. CONSUME the proto-blocks (take
    // them out, then `into_iter`) so each is freed as it's converted and the
    // lowered statements are MOVED, not cloned — this avoids holding the whole
    // program twice (the dominant memory cost on huge inlines like stock BearSSL).
    let protos = std::mem::take(&mut inl.proto);
    let mut blocks: Vec<Block> = Vec::with_capacity(protos.len());
    let mut pc: u32 = 1; // PC 0 is the virtual entry, matching the live path.
    for (i, pb) in protos.into_iter().enumerate() {
        let stmts = pb.stmts;
        let n = stmts.len();
        let mut body_stmts: Vec<Stmt> = Vec::with_capacity(n.saturating_sub(1));
        let mut terminator = Stmt::Return;
        let start_pc = pc;
        for (j, ps) in stmts.into_iter().enumerate() {
            let stmt = match ps {
                ProtoStmt::Lowered(s) => s,
                ProtoStmt::Goto(targets) => {
                    let ids: Vec<BlockId> = targets
                        .iter()
                        .map(|name| {
                            *label_to_block.get(name).unwrap_or_else(|| {
                                panic!("inline_lower: unknown goto target label: {}", name)
                            })
                        })
                        .collect();
                    Stmt::Goto { targets: ids }
                }
            };
            if j + 1 < n {
                body_stmts.push(stmt);
            } else {
                terminator = stmt;
            }
            pc += 1;
        }

        // assume_cond: first non-trivial assume among the leading statements.
        let mut assume_cond = None;
        for stmt in &body_stmts {
            match stmt {
                Stmt::Assert { .. } => continue,
                Stmt::Assume { expr } => {
                    assume_cond = Some(expr.clone());
                    break;
                }
                Stmt::AssumeTrue => continue,
                _ => break,
            }
        }

        blocks.push(Block {
            name: pb.name,
            id: i as BlockId,
            body: body_stmts,
            terminator,
            start_pc,
            assume_cond,
        });
    }

    // Drop blocks no execution can reach (post-inline DeadBlockEliminationPass),
    // before alloc resolution so we don't resolve sizes in pruned blocks.
    dead_block_elim(&mut blocks, &mut label_to_block);

    // Resolve $CurrAddr allocation sizes on the lowered IR.
    resolve_alloc_sizes(&mut blocks, inl.intern.names());

    let curr_addr_id = inl.intern.get("$CurrAddr");
    let curr_addr_shadow_id = inl.intern.get("$CurrAddr.shadow");
    let m0_id = inl.intern.get("$M.0");
    let m0_shadow_id = inl.intern.get("$M.0.shadow");

    // Resolve the compile-time static-scalar seeds (name→value, computed in
    // Python from the un-inlined program) to (VarId, value) and bake them in, so
    // the serialized package seeds them at init with no Python-AST native_meta.
    let mut baked_scalars: Vec<(VarId, i64)> = Vec::new();
    if let Some(d) = static_scalars {
        for (k, v) in d.iter() {
            let name: String = k.extract()?;
            let value: i64 = v.extract()?;
            if let Some(vid) = inl.intern.get(&name) {
                baked_scalars.push((vid, value));
            }
        }
    }

    super::normalize_is_external_assumes(&mut blocks);

    let n_blocks = blocks.len();
    Ok(CompiledProgram {
        blocks,
        label_to_block,
        var_names: inl.intern.names().to_vec(),
        name_to_var: std::mem::take(&mut inl.intern.map),
        entry_block: 0,
        entry_preconditions,
        mem_maps: inl.mem_maps,
        num_vars: inl.intern.len(),
        curr_addr_id,
        curr_addr_shadow_id,
        m0_id,
        m0_shadow_id,
        // Loop metadata is "no loop" (safe for a no_trace concrete run — the
        // tracer would emit iter_id=0; full loop analysis is a later refinement).
        loop_header_live_vars: FxHashMap::default(),
        is_loop_header: vec![false; n_blocks],
        block_innermost_header: vec![None; n_blocks],
        loop_parent_header: vec![None; n_blocks],
        static_scalars: baked_scalars,
        mode: inl.intern.mode,
    })
}

impl<'py> Inliner<'py> {
    /// Intern global vars/consts and record global memory maps.
    fn process_globals(&mut self, decls: &Bound<'py, PyList>) -> PyResult<()> {
        for decl in decls.iter() {
            let tn = decl.get_type().name()?.to_string();
            match tn.as_str() {
                "StorageDeclaration" | "VariableDeclaration" => {
                    self.intern_storage_decl(&decl)?;
                }
                "ConstantDeclaration" => {
                    let names = decl.getattr("names")?;
                    let names_list: &Bound<'_, PyList> = names.downcast()?;
                    for item in names_list.iter() {
                        let name: String = item.extract()?;
                        self.intern.intern(&name);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Intern a storage/variable/param/return declaration. Map-typed decls are
    /// registered as `MemMapInfo` under their (possibly frame-prefixed) name.
    /// This includes a callee's map PARAMETERS — e.g. memset/memcpy's
    /// `M.ret`/`M.src`, which become per-instance maps `inline$…$M.ret` bound to
    /// the caller's map by the arg/return copies (a map parameter's by-value
    /// semantics — the same shape Boogie produces).
    fn intern_storage_decl(&mut self, decl: &Bound<'py, PyAny>) -> PyResult<()> {
        let names = decl.getattr("names")?;
        let names_list: &Bound<'_, PyList> = names.downcast()?;
        let type_obj = decl.getattr("type")?;
        let is_map = type_obj.get_type().name()?.to_string() == "MapType";

        if is_map {
            let bw = convert_type_to_bitwidth(self.py, &type_obj)?;
            for item in names_list.iter() {
                let raw: String = item.extract()?;
                let var_id = self.intern.intern(&raw); // frame-aware prefixing
                let name = self.intern.names()[var_id as usize].clone();
                self.mem_maps.push(MemMapInfo {
                    name,
                    var_id,
                    index_bit_width: bw.0,
                    element_bit_width: bw.1,
                });
            }
        } else {
            for item in names_list.iter() {
                let name: String = item.extract()?;
                self.intern.intern(&name);
            }
        }
        Ok(())
    }

    /// Pre-intern an implementation's local declarations (so even unreferenced
    /// locals get a VarId, matching the live path). `frame` is installed for
    /// the duration so callee locals are renamed.
    fn process_impl_locals(
        &mut self,
        impl_decl: &Bound<'py, PyAny>,
        frame: Option<Frame>,
    ) -> PyResult<()> {
        let saved = self.intern.set_frame(frame);
        let body = impl_decl.getattr("body")?;
        let locals = body.getattr("locals")?;
        let locals_list: &Bound<'_, PyList> = locals.downcast()?;
        for local_decl in locals_list.iter() {
            let tn = local_decl.get_type().name()?.to_string();
            if tn == "StorageDeclaration" || tn == "VariableDeclaration" {
                self.intern_storage_decl(&local_decl)?;
            }
        }
        self.intern.set_frame(saved);
        Ok(())
    }

    /// Pre-intern an implementation's parameters and returns.
    fn process_impl_params(
        &mut self,
        impl_decl: &Bound<'py, PyAny>,
        frame: Option<Frame>,
    ) -> PyResult<()> {
        let saved = self.intern.set_frame(frame);
        for attr in ["parameters", "returns"] {
            let list = impl_decl.getattr(attr)?;
            let list: &Bound<'_, PyList> = list.downcast()?;
            for p in list.iter() {
                self.intern_storage_decl(&p)?;
            }
        }
        self.intern.set_frame(saved);
        Ok(())
    }

    fn start_block(&mut self, name: String) {
        debug_assert!(self.cur.is_none(), "start_block with an unflushed block");
        self.cur = Some(ProtoBlock {
            name,
            stmts: Vec::new(),
        });
    }

    fn push(&mut self, ps: ProtoStmt) {
        self.cur
            .as_mut()
            .expect("push with no current block")
            .stmts
            .push(ps);
    }

    fn flush_block(&mut self) {
        if let Some(mut pb) = self.cur.take() {
            // Const-fold each statement's expressions as the block is finalized,
            // so the accumulating proto stays smaller (cuts the build peak). This
            // is the post-inline ConstantFoldPass, run interleaved during inline.
            for ps in pb.stmts.iter_mut() {
                if let ProtoStmt::Lowered(s) = ps {
                    fold_stmt(s, self.intern.mode);
                }
            }
            self.proto.push(pb);
        }
    }

    /// True if the current block's last statement is (or const-folds to)
    /// `assume false`, making the block a dead-end. Cheap: folds only that one
    /// tail assume in place (flush_block re-folds the block idempotently later).
    /// Quantified/`$isExternal` assumes are other Stmt variants, so this only
    /// ever inspects a plain `Stmt::Assume`.
    fn cur_block_went_dead(&mut self) -> bool {
        match self.cur.as_mut().and_then(|b| b.stmts.last_mut()) {
            Some(ProtoStmt::Lowered(Stmt::Assume { expr })) => {
                fold_in_place(expr, self.intern.mode);
                matches!(expr, Expr::Bool(false))
            }
            _ => false,
        }
    }

    /// Inline-lower one implementation's blocks. `frame` renames the callee's
    /// locals/labels (None for the entry). `return_target`, when set, is the
    /// continuation block a `return` jumps to (None ⇒ emit `Return`).
    fn inline_impl(
        &mut self,
        impl_decl: &Bound<'py, PyAny>,
        frame: Option<Frame>,
        return_target: Option<String>,
    ) -> PyResult<()> {
        let saved = self.intern.set_frame(frame);

        let body = impl_decl.getattr("body")?;
        let blocks = body.getattr("blocks")?;
        let blocks_list: &Bound<'_, PyList> = blocks.downcast()?;

        for block in blocks_list.iter() {
            let raw_name: String = block.getattr("name")?.extract()?;
            let block_name = self.intern.apply_label(&raw_name);
            self.start_block(block_name);

            let stmts = block.getattr("statements")?;
            let stmts_list: &Bound<'_, PyList> = stmts.downcast()?;
            let n = stmts_list.len();
            let _ = n;
            for stmt in stmts_list.iter() {
                self.process_stmt(&stmt, return_target.as_deref())?;
                if self.cur_block_went_dead() {
                    // `assume false` (literal — SMACK's unresolved-indirect-call
                    // fallback — or one a local const-fold just produced): the
                    // rest of this block is unreachable, so stop processing it.
                    // We never inline the dead tail's calls (the build-peak
                    // lever); ensure_terminator caps the block, and the trailing
                    // goto it skips is dropped by dead_block_elim. Sound: the VM
                    // treats `assume false` as a failed assertion (vm.rs), so no
                    // execution reaches past it anyway.
                    self.dead_block_cutoffs += 1;
                    break;
                }
            }
            // A block whose source doesn't end in goto/return (a structured-If
            // exit, a trailing assume, a tail call) falls through to the proc's
            // implicit return — which is a goto to the continuation when inlined,
            // or `Return` at the entry.
            self.ensure_terminator(return_target.as_deref());
            self.flush_block();
        }

        self.intern.set_frame(saved);
        Ok(())
    }

    /// Append an implicit terminator if the current block doesn't already end in
    /// goto/return (fall-through ⇒ proc return ⇒ continuation-goto when inlined).
    fn ensure_terminator(&mut self, return_target: Option<&str>) {
        let terminated = matches!(
            self.cur.as_ref().and_then(|b| b.stmts.last()),
            Some(ProtoStmt::Goto(_)) | Some(ProtoStmt::Lowered(Stmt::Return))
        );
        if !terminated {
            match return_target {
                Some(t) => self.push(ProtoStmt::Goto(vec![t.to_string()])),
                None => self.push(ProtoStmt::Lowered(Stmt::Return)),
            }
        }
    }

    fn process_stmt(
        &mut self,
        stmt: &Bound<'py, PyAny>,
        return_target: Option<&str>,
    ) -> PyResult<()> {
        let tn = stmt.get_type().name()?.to_string();
        match tn.as_str() {
            "GotoStatement" => {
                let idents = stmt.getattr("identifiers")?;
                let idents_list: &Bound<'_, PyList> = idents.downcast()?;
                let targets: Vec<String> = idents_list
                    .iter()
                    .map(|id| {
                        let raw: String = id.getattr("name").unwrap().extract().unwrap();
                        self.intern.apply_label(&raw)
                    })
                    .collect();
                self.push(ProtoStmt::Goto(targets));
            }
            "ReturnStatement" => match return_target {
                Some(t) => self.push(ProtoStmt::Goto(vec![t.to_string()])),
                None => self.push(ProtoStmt::Lowered(Stmt::Return)),
            },
            "CallStatement" => {
                let proc = stmt.getattr("procedure")?;
                let callee: String = proc.getattr("name")?.extract()?;
                let inlinable = self.inline_names.contains(&callee)
                    && self.body_by_name.contains_key(&callee)
                    && !self.active.contains(&callee);
                if inlinable {
                    // A call may be the block's last statement (e.g. $alloc's tail
                    // call); inline_call flushes the pre-call block and opens a
                    // continuation, which ensure_terminator caps if nothing follows.
                    self.inline_call(stmt, &callee)?;
                } else {
                    if self.active.contains(&callee) && self.inline_names.contains(&callee) {
                        // Recursion: the callee is already on the inline stack.
                        // Boogie's {:inline N} (SMACK marks N=1) bounds recursion
                        // by inlining the first occurrence and leaving the
                        // recursive call as a residual havoc — our `active` set is
                        // exactly that bound. Count it (observability) and lower
                        // the cyclic call as a residual, matching Boogie.
                        self.bounded_recursions += 1;
                    }
                    // Residual call (printf/nondet/$alloc/bodiless/recursive/…).
                    let lowered = lower_call(self.py, stmt, &mut self.intern)?;
                    self.push(ProtoStmt::Lowered(lowered));
                }
            }
            _ => {
                let lowered = lower_stmt(self.py, stmt, &mut self.intern, &self.empty_labels)?;
                self.push(ProtoStmt::Lowered(lowered));
            }
        }
        Ok(())
    }

    /// Splice an inlinable call: bind params, jump to the callee entry, inline
    /// the callee body, then bind returns in a fresh continuation block.
    fn inline_call(&mut self, stmt: &Bound<'py, PyAny>, callee: &str) -> PyResult<()> {
        let n = self.counter;
        self.counter += 1;
        let prefix = format!("inline${}${}$", callee, n);
        if std::env::var("INLINE_DEBUG").is_ok() {
            eprintln!("[inline_lower] inlining call #{} to {}", n, callee);
        }

        let impl_decl = self.body_by_name.get(callee).unwrap().clone();
        let params = flatten_names(&impl_decl, "parameters")?;
        let returns = flatten_names(&impl_decl, "returns")?;
        let callee_locals = self.callee_local_set(&impl_decl)?;

        // in-param bindings: `inline$Q$N$param := arg` (args under caller frame).
        let args = stmt.getattr("arguments")?;
        let args_list: &Bound<'_, PyList> = args.downcast()?;
        assert_eq!(
            params.len(),
            args_list.len(),
            "inline_lower: arity mismatch calling `{}` ({} params, {} args)",
            callee,
            params.len(),
            args_list.len()
        );
        for (p, arg) in params.iter().zip(args_list.iter()) {
            let rhs = lower_expr(self.py, &arg, &mut self.intern)?;
            let lhs = self.intern.intern_raw(&format!("{}{}", prefix, p));
            self.push(ProtoStmt::Lowered(Stmt::Assign1 { lhs, rhs }));
        }

        // Jump to the callee's entry block, ending the current (pre-call) block.
        let entry_label = {
            let body = impl_decl.getattr("body")?;
            let blocks = body.getattr("blocks")?;
            let blocks_list: &Bound<'_, PyList> = blocks.downcast()?;
            let first = blocks_list.get_item(0)?;
            let raw: String = first.getattr("name")?.extract()?;
            format!("{}{}", prefix, raw)
        };
        let cont_label = format!("$inline_cont${}", n);
        self.push(ProtoStmt::Goto(vec![entry_label]));
        self.flush_block();

        // Inline the callee body under its frame; returns jump to cont_label.
        let frame = Frame {
            prefix: prefix.clone(),
            locals: callee_locals,
        };
        self.process_impl_locals(&impl_decl, Some(clone_frame(&frame)))?;
        self.process_impl_params(&impl_decl, Some(clone_frame(&frame)))?;
        self.active.insert(callee.to_string());
        self.inline_impl(&impl_decl, Some(frame), Some(cont_label.clone()))?;
        self.active.remove(callee);

        // Continuation: out-param bindings `target := inline$Q$N$return`.
        self.start_block(cont_label);
        let assigns = stmt.getattr("assignments")?;
        let assigns_list: &Bound<'_, PyList> = assigns.downcast()?;
        assert_eq!(
            returns.len(),
            assigns_list.len(),
            "inline_lower: return-arity mismatch calling `{}` ({} returns, {} targets)",
            callee,
            returns.len(),
            assigns_list.len()
        );
        for (r, target) in returns.iter().zip(assigns_list.iter()) {
            let tname: String = target.getattr("name")?.extract()?;
            let lhs = self.intern.intern(&tname); // caller frame active again
            let rhs = Expr::Var(self.intern.intern_raw(&format!("{}{}", prefix, r)));
            self.push(ProtoStmt::Lowered(Stmt::Assign1 { lhs, rhs }));
        }
        // The caller's remaining statements accumulate into this continuation.
        Ok(())
    }

    /// The set of a callee's local + param + return names (the names a frame
    /// renames).
    fn callee_local_set(&self, impl_decl: &Bound<'py, PyAny>) -> PyResult<FxHashSet<String>> {
        let mut set: FxHashSet<String> = FxHashSet::default();
        for n in flatten_names(impl_decl, "parameters")? {
            set.insert(n);
        }
        for n in flatten_names(impl_decl, "returns")? {
            set.insert(n);
        }
        let body = impl_decl.getattr("body")?;
        let locals = body.getattr("locals")?;
        let locals_list: &Bound<'_, PyList> = locals.downcast()?;
        for local_decl in locals_list.iter() {
            let names = local_decl.getattr("names")?;
            let names_list: &Bound<'_, PyList> = names.downcast()?;
            for item in names_list.iter() {
                let name: String = item.extract()?;
                set.insert(name);
            }
        }
        Ok(set)
    }
}

fn clone_frame(f: &Frame) -> Frame {
    Frame {
        prefix: f.prefix.clone(),
        locals: f.locals.clone(),
    }
}

/// Flatten the names of a proc/impl's `parameters` or `returns` list.
fn flatten_names(impl_decl: &Bound<'_, PyAny>, attr: &str) -> PyResult<Vec<String>> {
    let list = impl_decl.getattr(attr)?;
    let list: &Bound<'_, PyList> = list.downcast()?;
    let mut out = Vec::new();
    for decl in list.iter() {
        let names = decl.getattr("names")?;
        let names_list: &Bound<'_, PyList> = names.downcast()?;
        for item in names_list.iter() {
            out.push(item.extract::<String>()?);
        }
    }
    Ok(out)
}

/// Resolve `HavocCurrAddr { alloc_size_var: MAX }` placeholders by scanning each
/// block for the following `assume` that ties the havoc'd `$CurrAddr` to the
/// allocation size variable (a `$n…` var). Mirrors the post-pass in
/// `lower_program_full`, but on the lowered IR.
fn resolve_alloc_sizes(blocks: &mut [Block], var_names: &[String]) {
    for block in blocks.iter_mut() {
        resolve_alloc_in_stmts(&mut block.body, var_names);
    }
}

/// Resolve `HavocCurrAddr` placeholders in one statement list, then recurse into
/// nested `If`/`While` bodies — allocations from an inlined `$$alloc` live inside
/// a structured `If`, not at block top-level.
fn resolve_alloc_in_stmts(stmts: &mut Vec<Stmt>, var_names: &[String]) {
    let havocs: Vec<(usize, VarId)> = stmts
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match s {
            Stmt::HavocCurrAddr {
                var_id,
                alloc_size_var,
            } if *alloc_size_var == u32::MAX => Some((i, *var_id)),
            _ => None,
        })
        .collect();
    for (i, havoc_var) in havocs {
        let mut size_var: Option<VarId> = None;
        for s in stmts.iter().skip(i + 1) {
            match s {
                Stmt::Assume { expr } | Stmt::Assert { expr } => {
                    if expr_refs_var(expr, havoc_var) {
                        if let Some(v) = expr_find_var_named(expr, var_names, "$n") {
                            size_var = Some(v);
                            break;
                        }
                    }
                }
                Stmt::AssumeTrue | Stmt::HavocCurrAddr { .. } | Stmt::Havoc { .. } => continue,
                _ => break,
            }
        }
        if let Some(svid) = size_var {
            if let Stmt::HavocCurrAddr { alloc_size_var, .. } = &mut stmts[i] {
                *alloc_size_var = svid;
            }
        }
    }
    for s in stmts.iter_mut() {
        match s {
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                resolve_alloc_in_stmts(then_body, var_names);
                resolve_alloc_in_stmts(else_body, var_names);
            }
            Stmt::While { body, .. } => resolve_alloc_in_stmts(body, var_names),
            _ => {}
        }
    }
}

fn expr_refs_var(expr: &Expr, target: VarId) -> bool {
    match expr {
        Expr::Var(v) => *v == target,
        Expr::Const(_) | Expr::ConstBig(_) | Expr::Bool(_) | Expr::IsExternal => false,
        Expr::Not(e) => expr_refs_var(e, target),
        Expr::BinOp { lhs, rhs, .. } => expr_refs_var(lhs, target) || expr_refs_var(rhs, target),
        Expr::Builtin { args, .. } => args.iter().any(|a| expr_refs_var(a, target)),
        Expr::Store {
            map, index, value, ..
        } => {
            expr_refs_var(map, target)
                || expr_refs_var(index, target)
                || expr_refs_var(value, target)
        }
        Expr::Load { map, index, .. } => expr_refs_var(map, target) || expr_refs_var(index, target),
        Expr::IfThenElse { cond, then_, else_ } => {
            expr_refs_var(cond, target)
                || expr_refs_var(then_, target)
                || expr_refs_var(else_, target)
        }
    }
}

fn expr_find_var_named(expr: &Expr, var_names: &[String], substr: &str) -> Option<VarId> {
    match expr {
        Expr::Var(v) => {
            if var_names
                .get(*v as usize)
                .map(|n| n.contains(substr))
                .unwrap_or(false)
            {
                Some(*v)
            } else {
                None
            }
        }
        Expr::Const(_) | Expr::ConstBig(_) | Expr::Bool(_) | Expr::IsExternal => None,
        Expr::Not(e) => expr_find_var_named(e, var_names, substr),
        Expr::BinOp { lhs, rhs, .. } => expr_find_var_named(lhs, var_names, substr)
            .or_else(|| expr_find_var_named(rhs, var_names, substr)),
        Expr::Builtin { args, .. } => args
            .iter()
            .find_map(|a| expr_find_var_named(a, var_names, substr)),
        Expr::Store {
            map, index, value, ..
        } => expr_find_var_named(map, var_names, substr)
            .or_else(|| expr_find_var_named(index, var_names, substr))
            .or_else(|| expr_find_var_named(value, var_names, substr)),
        Expr::Load { map, index, .. } => expr_find_var_named(map, var_names, substr)
            .or_else(|| expr_find_var_named(index, var_names, substr)),
        Expr::IfThenElse { cond, then_, else_ } => expr_find_var_named(cond, var_names, substr)
            .or_else(|| expr_find_var_named(then_, var_names, substr))
            .or_else(|| expr_find_var_named(else_, var_names, substr)),
    }
}

// ---------------------------------------------------------------------------
// Constant folding (the post-inline ConstantFoldPass, run interleaved during
// inline so the proto stays small). Matches `vm::eval` EXACTLY — same BinOp
// rules and the same `builtins` ops — so the folded program executes
// identically (MASK_64 is all-ones, i.e. identity, so it's omitted).
// ---------------------------------------------------------------------------

/// Is this block a guaranteed dead-end — execution reaching it never flows past
/// it? True iff its body holds an `assume false` (SMACK's unresolved-indirect-
/// call fallback, or a const-folded false guard). The VM treats `assume false`
/// as a failed assertion (vm.rs), so such a block contributes no successor edges.
fn block_is_dead_end(blk: &Block) -> bool {
    blk.body
        .iter()
        .any(|s| matches!(s, Stmt::Assume { expr: Expr::Bool(false) }))
}

/// Post-inline dead-block elimination — the Python `DeadBlockEliminationPass`,
/// run on the assembled flat procedure. Drops blocks unreachable from the entry
/// over the goto graph (treating `assume false` blocks as dead-ends), then
/// reindexes `BlockId`s (== Vec position), remaps every surviving goto target,
/// and rebuilds `label_to_block`. Entry (block 0) is always reachable and keeps
/// index 0 (survivors retain order).
///
/// Soundness: only blocks that NO execution can reach are removed. Every target
/// of every reachable non-dead-end block is itself marked reachable, so a
/// surviving `goto`/`resolve_branch` never references a pruned block. A
/// dead-end's own trailing goto is semantically unreachable (the VM halts at the
/// `assume false`), so it is rewritten to `Return` rather than remapped.
fn dead_block_elim(blocks: &mut Vec<Block>, label_to_block: &mut FxHashMap<String, BlockId>) {
    let n = blocks.len();
    if n == 0 {
        return;
    }
    // 1. Reachability worklist from the entry; a dead-end block doesn't expand.
    let mut reachable = vec![false; n];
    reachable[0] = true;
    let mut stack = vec![0usize];
    while let Some(b) = stack.pop() {
        if block_is_dead_end(&blocks[b]) {
            continue;
        }
        if let Stmt::Goto { targets } = &blocks[b].terminator {
            for &t in targets {
                let t = t as usize;
                if t < n && !reachable[t] {
                    reachable[t] = true;
                    stack.push(t);
                }
            }
        }
    }
    let removed = reachable.iter().filter(|&&r| !r).count();
    if removed == 0 {
        return;
    }
    // 2. old index -> new index for survivors (order preserved; entry stays 0).
    let mut remap: Vec<Option<BlockId>> = vec![None; n];
    let mut next: BlockId = 0;
    for (old, &r) in reachable.iter().enumerate() {
        if r {
            remap[old] = Some(next);
            next += 1;
        }
    }
    // 3. Rebuild the block list with remapped ids + goto targets.
    let old_blocks = std::mem::take(blocks);
    let mut kept: Vec<Block> = Vec::with_capacity(next as usize);
    for (old, mut blk) in old_blocks.into_iter().enumerate() {
        if !reachable[old] {
            continue;
        }
        blk.id = remap[old].unwrap();
        if block_is_dead_end(&blk) {
            // The trailing goto is unreachable (VM halts at `assume false`);
            // drop the dangling edge so we needn't remap a possibly-pruned target.
            blk.terminator = Stmt::Return;
        } else if let Stmt::Goto { targets } = &mut blk.terminator {
            for t in targets.iter_mut() {
                *t = remap[*t as usize].expect("dead_block_elim: live goto target was pruned");
            }
        }
        kept.push(blk);
    }
    // 4. Rebuild the label index.
    label_to_block.clear();
    for blk in &kept {
        label_to_block.insert(blk.name.clone(), blk.id);
    }
    *blocks = kept;
    eprintln!(
        "[inline_lower] dead_block_elim: removed {}/{} unreachable blocks",
        removed, n
    );
}

