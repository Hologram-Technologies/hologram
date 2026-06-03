//! Bounded refinement execution over an addressed [`InferenceSession`].
//!
//! Refinement is an execution strategy, not a graph operation: each pass is a
//! normal compiled session run, and pass-to-pass state flows by content label.

use alloc::vec::Vec;

use hologram_archive::ContentLabel;
use smallvec::SmallVec;
use thiserror::Error;

use crate::buffer::InputBuffer;
use crate::error::ExecError;
use crate::session::{InferenceSession, SessionBackend};

const MAX_VALIDATORS: usize = 4;

/// Cost class for a validator in a refinement plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatorCost {
    /// Fixed metadata-only work.
    Constant,
    /// Work proportional to the number of state ports.
    StatePorts,
    /// Work proportional to the byte size of the state.
    StateBytes,
    /// Future compiled validator-plan work.
    CompiledPlan,
}

/// Built-in validator kinds for the first refinement strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatorKind {
    /// Accept when output labels exactly match input labels.
    StableLabels,
    /// Accept when resolved output bytes exactly match input bytes.
    StableBytes,
}

impl ValidatorKind {
    /// Return the validator's explicit cost class.
    #[must_use]
    pub const fn cost(self) -> ValidatorCost {
        match self {
            Self::StableLabels => ValidatorCost::StatePorts,
            Self::StableBytes => ValidatorCost::StateBytes,
        }
    }
}

/// Bounded repair behavior after normal pass convergence fails.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RepairPolicy {
    /// Do not run repair passes.
    #[default]
    None,
    /// Retry the same compiled pass a bounded number of extra times.
    RetryPass {
        /// Extra pass budget after `max_passes` is exhausted.
        extra_passes: u8,
    },
}

/// How a refinement run converged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceKind {
    /// State labels are stable.
    LabelStable,
    /// State bytes are stable.
    ByteStable,
    /// Multiple configured validators accepted the state.
    ValidatorAccepted,
}

/// Terminal status for a refinement run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementStatus {
    /// Normal pass budget reached an accepted state.
    Converged(ConvergenceKind),
    /// Repair pass budget reached an accepted state.
    Repaired(ConvergenceKind),
    /// Normal pass budget ended before validators accepted.
    PassBoundReached,
    /// Repair pass budget ended before validators accepted.
    RepairBoundReached,
}

/// Outcome of one validator in the final evaluated pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatorOutcome {
    /// Validator that produced this result.
    pub validator: ValidatorKind,
    /// Whether the validator accepted the transition.
    pub accepted: bool,
    /// Validator cost class.
    pub cost: ValidatorCost,
}

/// One state port in a refinement state contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementStatePort {
    /// Runtime dtype tag.
    pub dtype: u8,
    /// Number of elements in this state port.
    pub element_count: u64,
    /// Logical byte length for boundary and validator comparisons.
    pub byte_len: usize,
    /// Full row-major shape.
    pub shape: Vec<u64>,
}

/// Explicit state contract for planner-generated refinement plans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefinementStateContract {
    ports: Vec<RefinementStatePort>,
}

impl RefinementStateContract {
    /// Build a contract from planner-provided state ports.
    pub fn new(ports: Vec<RefinementStatePort>) -> Result<Self, RefinementError> {
        if ports.is_empty() {
            return Err(RefinementError::StateContractMismatch);
        }
        Ok(Self { ports })
    }

    /// Derive the contract from a session whose state ports already match.
    pub fn from_session<B: SessionBackend>(
        session: &InferenceSession<B>,
    ) -> Result<Self, RefinementError> {
        validate_implicit_session_state(session)?;
        Ok(Self {
            ports: session_input_contract(session),
        })
    }

    /// State ports in order.
    #[must_use]
    pub fn ports(&self) -> &[RefinementStatePort] {
        &self.ports
    }

    /// Number of state ports.
    #[must_use]
    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    fn validate_for<B: SessionBackend>(
        &self,
        session: &InferenceSession<B>,
    ) -> Result<(), RefinementError> {
        validate_contract_arity(self, session)?;
        validate_contract_ports(self, session)
    }
}

/// A compiled refinement strategy over a single session.
#[derive(Debug, Clone)]
pub struct RefinementPlan {
    max_passes: u8,
    validators: SmallVec<[ValidatorKind; MAX_VALIDATORS]>,
    repair_policy: RepairPolicy,
    state_contract: Option<RefinementStateContract>,
}

impl RefinementPlan {
    /// Start building a refinement plan.
    #[must_use]
    pub fn builder(max_passes: u8) -> RefinementPlanBuilder {
        RefinementPlanBuilder::new(max_passes)
    }

    /// Validate this plan against an existing session and return a runner.
    pub fn bind<'a, B: SessionBackend>(
        &'a self,
        session: &'a mut InferenceSession<B>,
    ) -> Result<RefinementRunner<'a, B>, RefinementError> {
        self.validate_for(session)?;
        Ok(RefinementRunner {
            session,
            plan: self,
        })
    }

    /// Execute this plan against an existing session from raw buffers.
    pub fn execute<B: SessionBackend>(
        &self,
        session: &mut InferenceSession<B>,
        inputs: &[InputBuffer<'_>],
    ) -> Result<RefinementReport, RefinementError> {
        let mut runner = self.bind(session)?;
        runner.execute(inputs)
    }

    /// Execute this plan against an existing session from resident labels.
    pub fn execute_addressed<B: SessionBackend>(
        &self,
        session: &mut InferenceSession<B>,
        initial: &[ContentLabel],
    ) -> Result<RefinementReport, RefinementError> {
        let mut runner = self.bind(session)?;
        runner.execute_addressed(initial)
    }

    /// Maximum normal pass count.
    #[must_use]
    pub const fn max_passes(&self) -> u8 {
        self.max_passes
    }

    /// Validators evaluated after each pass.
    #[must_use]
    pub fn validators(&self) -> &[ValidatorKind] {
        &self.validators
    }

    /// Repair policy evaluated after normal pass exhaustion.
    #[must_use]
    pub const fn repair_policy(&self) -> RepairPolicy {
        self.repair_policy
    }

    /// Explicit state contract, when supplied by a planner.
    #[must_use]
    pub fn state_contract(&self) -> Option<&RefinementStateContract> {
        self.state_contract.as_ref()
    }

    fn validate_for<B: SessionBackend>(
        &self,
        session: &InferenceSession<B>,
    ) -> Result<(), RefinementError> {
        validate_plan_shape(self)?;
        validate_session_state(self, session)?;
        Ok(())
    }
}

/// Builder for planner-produced refinement plans.
#[derive(Debug, Clone)]
pub struct RefinementPlanBuilder {
    max_passes: u8,
    validators: SmallVec<[ValidatorKind; MAX_VALIDATORS]>,
    repair_policy: RepairPolicy,
    state_contract: Option<RefinementStateContract>,
    too_many_validators: bool,
}

impl RefinementPlanBuilder {
    /// Create a builder with a bounded normal pass count.
    #[must_use]
    pub fn new(max_passes: u8) -> Self {
        Self {
            max_passes,
            validators: SmallVec::new(),
            repair_policy: RepairPolicy::None,
            state_contract: None,
            too_many_validators: false,
        }
    }

    /// Add one validator unless the bounded validator budget is exhausted.
    #[must_use]
    pub fn validator(mut self, validator: ValidatorKind) -> Self {
        if self.validators.len() < MAX_VALIDATORS {
            self.validators.push(validator);
        } else {
            self.too_many_validators = true;
        }
        self
    }

    /// Set the bounded repair policy.
    #[must_use]
    pub const fn repair_policy(mut self, policy: RepairPolicy) -> Self {
        self.repair_policy = policy;
        self
    }

    /// Attach an explicit state contract produced by a planner.
    #[must_use]
    pub fn state_contract(mut self, contract: RefinementStateContract) -> Self {
        self.state_contract = Some(contract);
        self
    }

    /// Build and validate the plan-local invariants.
    pub fn build(self) -> Result<RefinementPlan, RefinementError> {
        if self.too_many_validators {
            return Err(RefinementError::TooManyValidators);
        }
        let plan = self.into_plan();
        validate_plan_shape(&plan)?;
        Ok(plan)
    }

    fn into_plan(self) -> RefinementPlan {
        RefinementPlan {
            max_passes: self.max_passes,
            validators: self.validators,
            repair_policy: self.repair_policy,
            state_contract: self.state_contract,
        }
    }
}

/// Final report for a bounded refinement run.
#[derive(Debug, Clone)]
pub struct RefinementReport {
    /// Terminal run status.
    pub status: RefinementStatus,
    /// Total pass executions, including repair retries.
    pub passes: u8,
    /// Repair pass executions.
    pub repairs: u8,
    /// Final state labels in output-port order.
    pub final_labels: Vec<ContentLabel>,
    /// Aggregated kernel dispatch count across executed passes.
    pub dispatched: usize,
    /// Aggregated resident-reuse skip count across executed passes.
    pub skipped: usize,
    /// Final evaluated validator outcomes.
    pub validator_outcomes: SmallVec<[ValidatorOutcome; MAX_VALIDATORS]>,
    /// Distinct resident values after the run.
    pub resident_values: usize,
    /// Resident bytes after the run.
    pub resident_bytes: usize,
}

/// Errors raised before or during refinement execution.
#[derive(Debug, Error)]
pub enum RefinementError {
    /// The normal pass budget is zero.
    #[error("refinement plan has zero normal pass budget")]
    ZeroPassBudget,
    /// The plan contains no validators.
    #[error("refinement plan has no validators")]
    MissingValidator,
    /// More than the bounded validator budget was requested.
    #[error("refinement plan has too many validators")]
    TooManyValidators,
    /// The session cannot feed outputs back as the next state input.
    #[error("refinement state contract does not match the session ports")]
    StateContractMismatch,
    /// Initial state label count does not match the session inputs.
    #[error("refinement state arity mismatch")]
    StateArityMismatch,
    /// A state label needed for validation is not resident.
    #[error("refinement state bytes are not resident")]
    MissingStateBytes,
    /// The underlying session failed.
    #[error("refinement execution failed: {0}")]
    Execute(#[from] ExecError),
}

/// Borrowed refinement runner over an existing session.
pub struct RefinementRunner<'a, B: SessionBackend> {
    session: &'a mut InferenceSession<B>,
    plan: &'a RefinementPlan,
}

impl<B: SessionBackend> RefinementRunner<'_, B> {
    /// Execute refinement from raw boundary buffers.
    pub fn execute(
        &mut self,
        inputs: &[InputBuffer<'_>],
    ) -> Result<RefinementReport, RefinementError> {
        let labels = self.intern_inputs(inputs)?;
        self.execute_addressed(&labels)
    }

    /// Execute refinement from resident state labels.
    pub fn execute_addressed(
        &mut self,
        initial: &[ContentLabel],
    ) -> Result<RefinementReport, RefinementError> {
        self.validate_initial(initial)?;
        let mut state = initial.to_vec();
        let mut report = ReportState::default();
        if let Some(kind) = self.run_passes(&mut state, false, &mut report)? {
            return Ok(report.finish(RefinementStatus::Converged(kind), state, self.session));
        }
        self.run_repair(state, report)
    }

    /// Resolve a final state label through the borrowed session.
    #[must_use]
    pub fn resolve(&self, label: &ContentLabel) -> Option<&[u8]> {
        self.session.resolve(label)
    }

    /// Borrow the underlying inference session.
    #[must_use]
    pub fn session(&self) -> &InferenceSession<B> {
        self.session
    }

    fn intern_inputs(
        &mut self,
        inputs: &[InputBuffer<'_>],
    ) -> Result<Vec<ContentLabel>, RefinementError> {
        if inputs.len() != self.session.input_count() {
            return Err(RefinementError::StateArityMismatch);
        }
        let mut labels = Vec::with_capacity(inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            labels.push(self.intern_one(i, input.bytes)?);
        }
        Ok(labels)
    }

    fn intern_one(&mut self, index: usize, bytes: &[u8]) -> Result<ContentLabel, RefinementError> {
        let len = self.session.input_byte_len(index);
        if bytes.len() < len {
            return Err(RefinementError::StateContractMismatch);
        }
        Ok(self.session.intern_input(&bytes[..len]))
    }

    fn validate_initial(&self, initial: &[ContentLabel]) -> Result<(), RefinementError> {
        if initial.len() == self.session.input_count() {
            Ok(())
        } else {
            Err(RefinementError::StateArityMismatch)
        }
    }

    fn run_passes(
        &mut self,
        state: &mut Vec<ContentLabel>,
        repair: bool,
        report: &mut ReportState,
    ) -> Result<Option<ConvergenceKind>, RefinementError> {
        let limit = pass_limit(self.plan.repair_policy, repair, self.plan.max_passes);
        for _ in 0..limit {
            if let Some(kind) = self.run_one_pass(state, repair, report)? {
                return Ok(Some(kind));
            }
        }
        Ok(None)
    }

    fn run_one_pass(
        &mut self,
        state: &mut Vec<ContentLabel>,
        repair: bool,
        report: &mut ReportState,
    ) -> Result<Option<ConvergenceKind>, RefinementError> {
        let previous = core::mem::take(state);
        let next = self.execute_next(&previous, state)?;
        report.record_pass(repair, self.session);
        let outcome = self.validate_transition(&previous, &next, report)?;
        *state = next;
        Ok(outcome)
    }

    fn execute_next(
        &mut self,
        previous: &[ContentLabel],
        state: &mut Vec<ContentLabel>,
    ) -> Result<Vec<ContentLabel>, RefinementError> {
        match self.session.execute_addressed(previous) {
            Ok(next) => Ok(next),
            Err(err) => {
                state.extend_from_slice(previous);
                Err(err.into())
            }
        }
    }

    fn validate_transition(
        &self,
        previous: &[ContentLabel],
        next: &[ContentLabel],
        report: &mut ReportState,
    ) -> Result<Option<ConvergenceKind>, RefinementError> {
        report.validator_outcomes.clear();
        for validator in self.plan.validators() {
            let accepted = self.apply_validator(*validator, previous, next)?;
            report.push_outcome(*validator, accepted);
        }
        Ok(report.convergence_kind())
    }

    fn apply_validator(
        &self,
        validator: ValidatorKind,
        previous: &[ContentLabel],
        next: &[ContentLabel],
    ) -> Result<bool, RefinementError> {
        match validator {
            ValidatorKind::StableLabels => Ok(previous == next),
            ValidatorKind::StableBytes => self.bytes_stable(previous, next),
        }
    }

    fn bytes_stable(
        &self,
        previous: &[ContentLabel],
        next: &[ContentLabel],
    ) -> Result<bool, RefinementError> {
        for (i, (left, right)) in previous.iter().zip(next.iter()).enumerate() {
            if self.logical_bytes(left, i)? != self.logical_bytes(right, i)? {
                return Ok(false);
            }
        }
        Ok(previous.len() == next.len())
    }

    fn logical_bytes(&self, label: &ContentLabel, index: usize) -> Result<&[u8], RefinementError> {
        let len = self.session.input_byte_len(index);
        self.resolve_bytes(label)?
            .get(..len)
            .ok_or(RefinementError::MissingStateBytes)
    }

    fn resolve_bytes(&self, label: &ContentLabel) -> Result<&[u8], RefinementError> {
        self.session
            .resolve(label)
            .ok_or(RefinementError::MissingStateBytes)
    }

    fn run_repair(
        &mut self,
        mut state: Vec<ContentLabel>,
        mut report: ReportState,
    ) -> Result<RefinementReport, RefinementError> {
        if !has_repair_budget(self.plan.repair_policy) {
            return Ok(report.finish(RefinementStatus::PassBoundReached, state, self.session));
        }
        if let Some(kind) = self.run_passes(&mut state, true, &mut report)? {
            return Ok(report.finish(RefinementStatus::Repaired(kind), state, self.session));
        }
        Ok(report.finish(RefinementStatus::RepairBoundReached, state, self.session))
    }
}

/// Owning refinement wrapper for a preloaded session and plan.
pub struct CompiledRefinement<B: SessionBackend> {
    session: InferenceSession<B>,
    plan: RefinementPlan,
}

impl<B: SessionBackend> CompiledRefinement<B> {
    /// Validate and bind a session to a refinement plan.
    pub fn new(
        session: InferenceSession<B>,
        plan: RefinementPlan,
    ) -> Result<Self, RefinementError> {
        plan.validate_for(&session)?;
        Ok(Self { session, plan })
    }

    /// Execute refinement from raw boundary buffers.
    pub fn execute(
        &mut self,
        inputs: &[InputBuffer<'_>],
    ) -> Result<RefinementReport, RefinementError> {
        self.plan.execute(&mut self.session, inputs)
    }

    /// Execute refinement from resident state labels.
    pub fn execute_addressed(
        &mut self,
        initial: &[ContentLabel],
    ) -> Result<RefinementReport, RefinementError> {
        self.plan.execute_addressed(&mut self.session, initial)
    }

    /// Resolve a final state label through the wrapped session.
    #[must_use]
    pub fn resolve(&self, label: &ContentLabel) -> Option<&[u8]> {
        self.session.resolve(label)
    }

    /// Borrow the wrapped inference session.
    #[must_use]
    pub const fn session(&self) -> &InferenceSession<B> {
        &self.session
    }

    /// Borrow the wrapped refinement plan.
    #[must_use]
    pub const fn plan(&self) -> &RefinementPlan {
        &self.plan
    }

    /// Consume the wrapper and return the session.
    #[must_use]
    pub fn into_session(self) -> InferenceSession<B> {
        self.session
    }
}

#[derive(Default)]
struct ReportState {
    passes: u8,
    repairs: u8,
    dispatched: usize,
    skipped: usize,
    validator_outcomes: SmallVec<[ValidatorOutcome; MAX_VALIDATORS]>,
}

impl ReportState {
    fn record_pass<B: SessionBackend>(&mut self, repair: bool, session: &InferenceSession<B>) {
        self.passes = self.passes.saturating_add(1);
        self.repairs = self.repairs.saturating_add(u8::from(repair));
        self.dispatched = self.dispatched.saturating_add(session.last_dispatched());
        self.skipped = self.skipped.saturating_add(session.last_skipped());
    }

    fn push_outcome(&mut self, validator: ValidatorKind, accepted: bool) {
        self.validator_outcomes.push(ValidatorOutcome {
            validator,
            accepted,
            cost: validator.cost(),
        });
    }

    fn convergence_kind(&self) -> Option<ConvergenceKind> {
        if self.validator_outcomes.iter().any(|o| !o.accepted) {
            return None;
        }
        self.single_convergence_kind()
    }

    fn single_convergence_kind(&self) -> Option<ConvergenceKind> {
        match self.validator_outcomes.as_slice() {
            [one] => Some(kind_for_validator(one.validator)),
            [] => None,
            _ => Some(ConvergenceKind::ValidatorAccepted),
        }
    }

    fn finish<B: SessionBackend>(
        self,
        status: RefinementStatus,
        final_labels: Vec<ContentLabel>,
        session: &InferenceSession<B>,
    ) -> RefinementReport {
        RefinementReport {
            status,
            passes: self.passes,
            repairs: self.repairs,
            final_labels,
            dispatched: self.dispatched,
            skipped: self.skipped,
            validator_outcomes: self.validator_outcomes,
            resident_values: session.resident_count(),
            resident_bytes: session.resident_bytes(),
        }
    }
}

fn validate_plan_shape(plan: &RefinementPlan) -> Result<(), RefinementError> {
    if plan.max_passes == 0 {
        return Err(RefinementError::ZeroPassBudget);
    }
    if plan.validators.is_empty() {
        return Err(RefinementError::MissingValidator);
    }
    Ok(())
}

fn validate_session_state<B: SessionBackend>(
    plan: &RefinementPlan,
    session: &InferenceSession<B>,
) -> Result<(), RefinementError> {
    match plan.state_contract() {
        Some(contract) => contract.validate_for(session),
        None => validate_implicit_session_state(session),
    }
}

fn validate_implicit_session_state<B: SessionBackend>(
    session: &InferenceSession<B>,
) -> Result<(), RefinementError> {
    validate_equal_arity(session)?;
    validate_matching_session_ports(session)
}

fn validate_equal_arity<B: SessionBackend>(
    session: &InferenceSession<B>,
) -> Result<(), RefinementError> {
    if session.input_count() == session.output_count() {
        Ok(())
    } else {
        Err(RefinementError::StateContractMismatch)
    }
}

fn validate_matching_session_ports<B: SessionBackend>(
    session: &InferenceSession<B>,
) -> Result<(), RefinementError> {
    for i in 0..session.input_count() {
        if !same_state_port(session, i) {
            return Err(RefinementError::StateContractMismatch);
        }
    }
    Ok(())
}

fn validate_contract_arity<B: SessionBackend>(
    contract: &RefinementStateContract,
    session: &InferenceSession<B>,
) -> Result<(), RefinementError> {
    if contract.port_count() == session.input_count()
        && contract.port_count() == session.output_count()
    {
        Ok(())
    } else {
        Err(RefinementError::StateContractMismatch)
    }
}

fn validate_contract_ports<B: SessionBackend>(
    contract: &RefinementStateContract,
    session: &InferenceSession<B>,
) -> Result<(), RefinementError> {
    for i in 0..contract.port_count() {
        if !contract_matches_session_port(contract, session, i) {
            return Err(RefinementError::StateContractMismatch);
        }
    }
    Ok(())
}

fn contract_matches_session_port<B: SessionBackend>(
    contract: &RefinementStateContract,
    session: &InferenceSession<B>,
    index: usize,
) -> bool {
    let expected = &contract.ports()[index];
    expected.matches_session_input(session, index)
        && expected.matches_session_output(session, index)
}

fn same_state_port<B: SessionBackend>(session: &InferenceSession<B>, index: usize) -> bool {
    let input = &session.input_ports()[index];
    let output = &session.output_ports()[index];
    input.dtype == output.dtype
        && input.element_count == output.element_count
        && input.shape == output.shape
        && session.input_byte_len(index) == session.output_byte_len(index)
}

fn session_input_contract<B: SessionBackend>(
    session: &InferenceSession<B>,
) -> Vec<RefinementStatePort> {
    let mut ports = Vec::with_capacity(session.input_count());
    for i in 0..session.input_count() {
        ports.push(RefinementStatePort::from_session_input(session, i));
    }
    ports
}

impl RefinementStatePort {
    fn from_session_input<B: SessionBackend>(session: &InferenceSession<B>, index: usize) -> Self {
        let port = &session.input_ports()[index];
        Self {
            dtype: port.dtype,
            element_count: port.element_count,
            byte_len: session.input_byte_len(index),
            shape: port.shape.clone(),
        }
    }

    fn matches_session_input<B: SessionBackend>(
        &self,
        session: &InferenceSession<B>,
        index: usize,
    ) -> bool {
        self.matches_port(
            session.input_ports()[index].dtype,
            session.input_ports()[index].element_count,
            &session.input_ports()[index].shape,
            session.input_byte_len(index),
        )
    }

    fn matches_session_output<B: SessionBackend>(
        &self,
        session: &InferenceSession<B>,
        index: usize,
    ) -> bool {
        self.matches_port(
            session.output_ports()[index].dtype,
            session.output_ports()[index].element_count,
            &session.output_ports()[index].shape,
            session.output_byte_len(index),
        )
    }

    fn matches_port(&self, dtype: u8, element_count: u64, shape: &[u64], byte_len: usize) -> bool {
        self.dtype == dtype
            && self.element_count == element_count
            && self.shape == shape
            && self.byte_len == byte_len
    }
}

const fn pass_limit(policy: RepairPolicy, repair: bool, max_passes: u8) -> u8 {
    match (repair, policy) {
        (false, _) => max_passes,
        (true, RepairPolicy::RetryPass { extra_passes }) => extra_passes,
        (true, RepairPolicy::None) => 0,
    }
}

const fn has_repair_budget(policy: RepairPolicy) -> bool {
    matches!(policy, RepairPolicy::RetryPass { extra_passes } if extra_passes > 0)
}

const fn kind_for_validator(validator: ValidatorKind) -> ConvergenceKind {
    match validator {
        ValidatorKind::StableLabels => ConvergenceKind::LabelStable,
        ValidatorKind::StableBytes => ConvergenceKind::ByteStable,
    }
}
