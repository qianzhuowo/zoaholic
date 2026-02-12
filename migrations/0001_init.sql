-- Zoaholic for Cloudflare D1
-- Initial schema (idempotent)

CREATE TABLE IF NOT EXISTS request_stats (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT,
  endpoint TEXT,
  client_ip TEXT,
  process_time REAL,
  first_response_time REAL,
  content_start_time REAL,
  provider TEXT,
  model TEXT,
  api_key TEXT,
  success INTEGER DEFAULT 0,
  status_code INTEGER,
  is_flagged INTEGER DEFAULT 0,
  text TEXT,
  prompt_tokens INTEGER DEFAULT 0,
  completion_tokens INTEGER DEFAULT 0,
  total_tokens INTEGER DEFAULT 0,
  prompt_price REAL DEFAULT 0.0,
  completion_price REAL DEFAULT 0.0,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  provider_id TEXT,
  provider_key_index INTEGER,
  api_key_name TEXT,
  api_key_group TEXT,
  retry_count INTEGER DEFAULT 0,
  retry_path TEXT,
  request_headers TEXT,
  request_body TEXT,
  upstream_request_headers TEXT,
  upstream_request_body TEXT,
  upstream_response_body TEXT,
  response_body TEXT,
  raw_data_expires_at DATETIME
);

CREATE TABLE IF NOT EXISTS channel_stats (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT,
  provider TEXT,
  model TEXT,
  api_key TEXT,
  provider_api_key TEXT,
  success INTEGER DEFAULT 0,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS admin_user (
  id INTEGER PRIMARY KEY,
  username TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  jwt_secret TEXT
);

CREATE TABLE IF NOT EXISTS app_config (
  id INTEGER PRIMARY KEY,
  config_json TEXT,
  config_yaml TEXT,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_request_stats_provider ON request_stats(provider);
CREATE INDEX IF NOT EXISTS idx_request_stats_model ON request_stats(model);
CREATE INDEX IF NOT EXISTS idx_request_stats_api_key ON request_stats(api_key);
CREATE INDEX IF NOT EXISTS idx_request_stats_success ON request_stats(success);
CREATE INDEX IF NOT EXISTS idx_request_stats_status_code ON request_stats(status_code);
CREATE INDEX IF NOT EXISTS idx_request_stats_timestamp ON request_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_request_stats_provider_id ON request_stats(provider_id);

CREATE INDEX IF NOT EXISTS idx_channel_stats_provider ON channel_stats(provider);
CREATE INDEX IF NOT EXISTS idx_channel_stats_model ON channel_stats(model);
CREATE INDEX IF NOT EXISTS idx_channel_stats_provider_api_key ON channel_stats(provider_api_key);
CREATE INDEX IF NOT EXISTS idx_channel_stats_timestamp ON channel_stats(timestamp);

CREATE INDEX IF NOT EXISTS idx_admin_user_username ON admin_user(username);
CREATE INDEX IF NOT EXISTS idx_app_config_updated_at ON app_config(updated_at);
