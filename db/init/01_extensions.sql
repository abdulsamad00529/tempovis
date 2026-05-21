-- Postgres init: enable extensions before Alembic runs migrations.
-- This file is executed by the postgres:16-alpine entrypoint on first boot.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- for future full-text search on summaries
