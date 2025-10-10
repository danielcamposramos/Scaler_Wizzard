# Multi-Vibe Human Contract

## Purpose
Guarantee that every automated action honours Daniel Ramos' authority as visionary architect. The contract must be acknowledged before any scaling run begins.

## Contract Fields
- `contract_version`: semantic version of this document.  
- `authorized_by`: human approver name (default: Daniel Ramos).  
- `approved_at`: ISO datetime stamped when the cockpit consent is given.  
- `session_id`: identifier tying consent to a specific scaling job.  
- `scope`: textual summary of the intended action (model, dataset, duration).  
- `rollback_plan`: location of the checkpoint to revert to if safety triggers fire.  
- `signoff`: explicit `"I accept responsibility for this run"` string supplied by the human.

## Runtime Enforcement
1. Pipeline constructor loads `config/human_contract.yaml` (generated via cockpit UI).  
2. Contract version is compared against `contract_version` embedded in runtime (`1.0.0`).  
3. Version mismatch or missing signoff raises `ContractNotAcceptedError` and aborts execution.  
4. Accepted contracts are persisted alongside telemetry for audit.

## Amendment Process
- Revise this document and bump `contract_version`.  
- Update cockpit UI to request re-confirmation from Daniel.  
- Store previous versions under `components/safety/archive/`.

## Outstanding Questions
- Should multiple human approvers be supported for collaborative sessions?  
- How do we surface contract expiry or stale approvals in the cockpit?  
- What legal phrasing best communicates responsibilities in future public releases?
