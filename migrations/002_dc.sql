-- Domain model. Names follow the blueprint's schema; additions are marked (+). Positions are row and column in the hall, not 3D.

CREATE TABLE facility (facility_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, external_ref text, name text, metadata jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(), seed int NOT NULL, clock timestamptz NOT NULL, UNIQUE (tenant_id, external_ref));   -- (+) seed of the synthetic hall, (+) clock: the end of its recorded history
SELECT enable_tenant_rls('facility');

CREATE TABLE room (room_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, facility_id uuid REFERENCES facility, status text NOT NULL, attributes jsonb NOT NULL DEFAULT '{}', observed_at timestamptz, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('room');

CREATE TABLE rack (rack_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, room_id uuid NOT NULL REFERENCES room, rack_no int NOT NULL, label text NOT NULL, row_no int NOT NULL, col_no int NOT NULL, aisle int NOT NULL, power_cap_kw real NOT NULL, gpu_slots int NOT NULL, UNIQUE (room_id, rack_no));
SELECT enable_tenant_rls('rack');

-- kind 'cpu' is the rack's CPU and storage gear as one PDU branch; each 'gpu' row is one 8-GPU training server.
CREATE TABLE server (server_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, rack_id uuid NOT NULL REFERENCES rack, kind text NOT NULL CHECK (kind IN ('cpu', 'gpu')), rated_kw real NOT NULL, workload_id uuid, power_cap real CHECK (power_cap > 0 AND power_cap <= 1), created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('server');
CREATE INDEX server_rack ON server (rack_id);

CREATE TABLE cooling_unit (unit_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, room_id uuid NOT NULL REFERENCES room, unit_no int NOT NULL, name text NOT NULL, aisle int NOT NULL, position_x real NOT NULL, airflow_kw real NOT NULL, status text NOT NULL CHECK (status IN ('running', 'failed')), setpoint_c real NOT NULL CHECK (setpoint_c BETWEEN 10 AND 30));
SELECT enable_tenant_rls('cooling_unit');

-- (+) facility_id; (+) unit_running: a CRAC's status, 1 or 0
CREATE TABLE sensor (sensor_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, external_ref text NOT NULL, kind text NOT NULL CHECK (kind IN ('inlet_temp', 'cpu_power', 'gpu_power', 'supply_temp', 'unit_running')), facility_id uuid NOT NULL REFERENCES facility, rack_id uuid REFERENCES rack, unit_id uuid REFERENCES cooling_unit, UNIQUE (tenant_id, external_ref));
SELECT enable_tenant_rls('sensor');

-- (+) Raw readings. Hourly here; at production rates this is a time-partitioned table or a time-series store, not OLTP.
CREATE TABLE telemetry (sensor_id uuid NOT NULL REFERENCES sensor, at timestamptz NOT NULL, value real NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (sensor_id, at));
SELECT enable_tenant_rls('telemetry');

CREATE TABLE workload (workload_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, facility_id uuid NOT NULL REFERENCES facility, name text NOT NULL, servers int NOT NULL CHECK (servers > 0), kw_per_server real NOT NULL, priority int NOT NULL DEFAULT 1, status text NOT NULL CHECK (status IN ('queued', 'placed')), created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('workload');

CREATE TABLE model_artifact (model_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, model_name text NOT NULL, version text NOT NULL, artifact_uri text NOT NULL, feature_schema jsonb NOT NULL, metrics jsonb NOT NULL DEFAULT '{}', approved boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now(), facility_id uuid NOT NULL REFERENCES facility, params jsonb NOT NULL, UNIQUE (tenant_id, model_name, version));   -- (+) facility, params
SELECT enable_tenant_rls('model_artifact');

-- The blueprint's decision record: every placement, cooling plan and failure response with its model, inputs hash, confidence and policy.
CREATE TABLE decision_record (decision_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, subject_type text NOT NULL, subject_id uuid NOT NULL, decision_type text NOT NULL CHECK (decision_type IN ('placement', 'cooling_plan', 'failure_response')), model_id uuid REFERENCES model_artifact, inputs_hash text NOT NULL, result jsonb NOT NULL, confidence double precision, policy_state jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(), created_by uuid NOT NULL);   -- (+) created_by
SELECT enable_tenant_rls('decision_record');

CREATE TABLE placement (decision_id uuid NOT NULL REFERENCES decision_record, workload_id uuid NOT NULL REFERENCES workload, rack_id uuid NOT NULL REFERENCES rack, servers int NOT NULL CHECK (servers > 0), tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (decision_id, workload_id, rack_id));
SELECT enable_tenant_rls('placement');

-- Predicted and measured inlet per rack for a decision (the evidence behind it).
CREATE TABLE thermal_state (decision_id uuid NOT NULL REFERENCES decision_record, rack_id uuid NOT NULL REFERENCES rack, predicted_c real NOT NULL, measured_c real, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (decision_id, rack_id));
SELECT enable_tenant_rls('thermal_state');

CREATE TABLE control_action (action_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, decision_id uuid NOT NULL REFERENCES decision_record, kind text NOT NULL CHECK (kind IN ('setpoint', 'power_cap')), target text NOT NULL, value real NOT NULL, previous real, mode text NOT NULL CHECK (mode IN ('automatic', 'needs_approval')), status text NOT NULL CHECK (status IN ('applied', 'pending_approval')), approved_by uuid, applied_at timestamptz, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('control_action');

CREATE TABLE energy_price (facility_id uuid NOT NULL REFERENCES facility, at timestamptz NOT NULL, usd_per_mwh real NOT NULL, kg_co2_per_mwh real NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (facility_id, at));
SELECT enable_tenant_rls('energy_price');
