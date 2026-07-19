//! Bounded refinement execution over compiled sessions.

use hologram_compiler::{compile, BackendKind};
use hologram_compute::CpuBackend;
use hologram_exec::{
    BufferArena, CompiledRefinement, ConvergenceKind, InferenceSession, InputBuffer,
    RefinementError, RefinementPlan, RefinementStateContract, RefinementStatus, RepairPolicy,
    ValidatorKind,
};
use hologram_graph::{
    node::Node,
    registry::{DTypeId, ShapeDescriptor, ShapeId},
    Graph, GraphOp, InputSource, OpKind,
};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

type TestSession = InferenceSession<CpuBackend<BufferArena>>;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn session_for(op: Option<OpKind>) -> TestSession {
    let compiled = compile(state_graph(op), BackendKind::Cpu, WittLevel::W32).unwrap();
    InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap()
}

fn state_graph(op: Option<OpKind>) -> Graph {
    let mut graph = Graph::new();
    let shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(2));
    let input = graph.add_node(state_node(GraphOp::Input, SmallVec::new(), shape));
    graph.add_input(input);
    let source = match op {
        Some(kind) => graph.add_node(state_node(
            GraphOp::Op(kind),
            SmallVec::from_iter([InputSource::Node(input)]),
            shape,
        )),
        None => input,
    };
    let output = graph.add_node(state_node(
        GraphOp::Output,
        SmallVec::from_iter([InputSource::Node(source)]),
        shape,
    ));
    graph.add_output(output);
    graph
}

fn state_node(op: GraphOp, inputs: SmallVec<[InputSource; 4]>, shape: ShapeId) -> Node {
    Node {
        op,
        inputs,
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    }
}

fn stable_bytes_plan(max_passes: u8) -> RefinementPlan {
    RefinementPlan::builder(max_passes)
        .validator(ValidatorKind::StableBytes)
        .build()
        .unwrap()
}

fn stable_labels_plan(max_passes: u8) -> RefinementPlan {
    RefinementPlan::builder(max_passes)
        .validator(ValidatorKind::StableLabels)
        .build()
        .unwrap()
}

#[test]
fn successful_label_convergence_on_identity_graph() {
    let session = session_for(None);
    let mut refinement = CompiledRefinement::new(session, stable_labels_plan(1)).unwrap();
    let input = f32_bytes(&[1.0, 2.0]);

    let report = refinement
        .execute(&[InputBuffer { bytes: &input }])
        .unwrap();

    assert_eq!(
        report.status,
        RefinementStatus::Converged(ConvergenceKind::LabelStable)
    );
    assert_eq!(report.passes, 1);
    assert_eq!(report.repairs, 0);
}

#[test]
fn successful_byte_convergence_after_two_relu_passes() {
    let session = session_for(Some(OpKind::Relu));
    let mut refinement = CompiledRefinement::new(session, stable_bytes_plan(2)).unwrap();
    let input = f32_bytes(&[-1.0, 2.0]);

    let report = refinement
        .execute(&[InputBuffer { bytes: &input }])
        .unwrap();

    assert_eq!(
        report.status,
        RefinementStatus::Converged(ConvergenceKind::ByteStable)
    );
    assert_eq!(report.passes, 2);
    assert_eq!(report.repairs, 0);
}

#[test]
fn validation_failure_returns_pass_bound_status() {
    let session = session_for(Some(OpKind::Relu));
    let mut refinement = CompiledRefinement::new(session, stable_bytes_plan(1)).unwrap();
    let input = f32_bytes(&[-1.0, 2.0]);

    let report = refinement
        .execute(&[InputBuffer { bytes: &input }])
        .unwrap();

    assert_eq!(report.status, RefinementStatus::PassBoundReached);
    assert_eq!(report.passes, 1);
    assert!(report.validator_outcomes.iter().any(|out| !out.accepted));
}

#[test]
fn repair_flow_uses_bounded_retry_pass() {
    let plan = RefinementPlan::builder(1)
        .validator(ValidatorKind::StableBytes)
        .repair_policy(RepairPolicy::RetryPass { extra_passes: 1 })
        .build()
        .unwrap();
    let session = session_for(Some(OpKind::Relu));
    let mut refinement = CompiledRefinement::new(session, plan).unwrap();
    let input = f32_bytes(&[-1.0, 2.0]);

    let report = refinement
        .execute(&[InputBuffer { bytes: &input }])
        .unwrap();

    assert_eq!(
        report.status,
        RefinementStatus::Repaired(ConvergenceKind::ByteStable)
    );
    assert_eq!(report.passes, 2);
    assert_eq!(report.repairs, 1);
}

#[test]
fn plan_executes_against_borrowed_session_without_taking_ownership() {
    let mut session = session_for(Some(OpKind::Relu));
    let plan = stable_bytes_plan(2);
    let input = f32_bytes(&[-1.0, 2.0]);

    let report = plan
        .execute(&mut session, &[InputBuffer { bytes: &input }])
        .unwrap();
    let outputs = session.execute(&[InputBuffer { bytes: &input }]).unwrap();

    assert_eq!(
        report.status,
        RefinementStatus::Converged(ConvergenceKind::ByteStable)
    );
    assert_eq!(outputs.len(), session.output_count());
}

#[test]
fn explicit_state_contract_matches_session() {
    let mut session = session_for(Some(OpKind::Relu));
    let contract = RefinementStateContract::from_session(&session).unwrap();
    let plan = RefinementPlan::builder(2)
        .validator(ValidatorKind::StableBytes)
        .state_contract(contract)
        .build()
        .unwrap();
    let input = f32_bytes(&[-1.0, 2.0]);

    let report = plan
        .execute(&mut session, &[InputBuffer { bytes: &input }])
        .unwrap();

    assert_eq!(
        report.status,
        RefinementStatus::Converged(ConvergenceKind::ByteStable)
    );
}

#[test]
fn explicit_state_contract_rejects_shape_mismatch() {
    let mut session = session_for(Some(OpKind::Relu));
    let contract = mutated_contract(&session, |port| port.shape = vec![1, 2]);
    let plan = plan_with_contract(contract);

    let result = plan.execute(&mut session, &[InputBuffer { bytes: &[] }]);

    assert_contract_rejected_without_dispatch(result, &session);
}

#[test]
fn explicit_state_contract_rejects_dtype_mismatch() {
    let mut session = session_for(Some(OpKind::Relu));
    let contract = mutated_contract(&session, |port| port.dtype = 7);
    let plan = plan_with_contract(contract);

    let result = plan.execute(&mut session, &[InputBuffer { bytes: &[] }]);

    assert_contract_rejected_without_dispatch(result, &session);
}

#[test]
fn explicit_state_contract_rejects_byte_len_mismatch() {
    let mut session = session_for(Some(OpKind::Relu));
    let contract = mutated_contract(&session, |port| port.byte_len += 4);
    let plan = plan_with_contract(contract);

    let result = plan.execute(&mut session, &[InputBuffer { bytes: &[] }]);

    assert_contract_rejected_without_dispatch(result, &session);
}

fn mutated_contract<F>(session: &TestSession, mutate: F) -> RefinementStateContract
where
    F: FnOnce(&mut hologram_exec::RefinementStatePort),
{
    let contract = RefinementStateContract::from_session(session).unwrap();
    let mut ports = contract.ports().to_vec();
    mutate(&mut ports[0]);
    RefinementStateContract::new(ports).unwrap()
}

fn plan_with_contract(contract: RefinementStateContract) -> RefinementPlan {
    RefinementPlan::builder(1)
        .validator(ValidatorKind::StableBytes)
        .state_contract(contract)
        .build()
        .unwrap()
}

fn assert_contract_rejected_without_dispatch(
    result: Result<hologram_exec::RefinementReport, RefinementError>,
    session: &TestSession,
) {
    assert!(matches!(
        result,
        Err(RefinementError::StateContractMismatch)
    ));
    assert_eq!(session.last_dispatched(), 0);
}

#[test]
fn planner_generated_plan_runs_deterministically() {
    let input = f32_bytes(&[-1.0, 2.0]);
    let report_a = run_relu_refinement(&input);
    let report_b = run_relu_refinement(&input);

    assert_eq!(report_a.status, report_b.status);
    assert_eq!(report_a.final_labels, report_b.final_labels);
    assert_eq!(report_a.passes, report_b.passes);
}

fn run_relu_refinement(input: &[u8]) -> hologram_exec::RefinementReport {
    let plan = RefinementPlan::builder(2)
        .validator(ValidatorKind::StableBytes)
        .build()
        .unwrap();
    let session = session_for(Some(OpKind::Relu));
    let mut refinement = CompiledRefinement::new(session, plan).unwrap();
    refinement.execute(&[InputBuffer { bytes: input }]).unwrap()
}

#[test]
fn invalid_plan_is_rejected() {
    let missing = RefinementPlan::builder(1).build();
    assert!(matches!(missing, Err(RefinementError::MissingValidator)));

    let zero = RefinementPlan::builder(0)
        .validator(ValidatorKind::StableBytes)
        .build();
    assert!(matches!(zero, Err(RefinementError::ZeroPassBudget)));
}

#[test]
fn initial_state_arity_mismatch_is_rejected() {
    let session = session_for(Some(OpKind::Relu));
    let mut refinement = CompiledRefinement::new(session, stable_bytes_plan(1)).unwrap();

    let result = refinement.execute_addressed(&[]);

    assert!(matches!(result, Err(RefinementError::StateArityMismatch)));
}

#[test]
fn refinement_does_not_mutate_loaded_graph_schedule() {
    let session = session_for(Some(OpKind::Relu));
    let kernel_count = session.kernel_count();
    let mut refinement = CompiledRefinement::new(session, stable_bytes_plan(2)).unwrap();
    let input = f32_bytes(&[-1.0, 2.0]);

    let _ = refinement
        .execute(&[InputBuffer { bytes: &input }])
        .unwrap();

    assert_eq!(refinement.session().kernel_count(), kernel_count);
}

#[test]
fn repeated_refinement_keeps_resident_memory_bounded() {
    let session = session_for(Some(OpKind::Relu));
    let mut refinement = CompiledRefinement::new(session, stable_bytes_plan(2)).unwrap();
    let input = f32_bytes(&[-1.0, 2.0]);
    let first = refinement
        .execute(&[InputBuffer { bytes: &input }])
        .unwrap();

    for _ in 0..32 {
        let report = refinement
            .execute(&[InputBuffer { bytes: &input }])
            .unwrap();
        assert!(report.resident_bytes <= first.resident_bytes.saturating_mul(2));
    }
}
