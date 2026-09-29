"""service schema: invitations, batches, documents, predictions, audit log, Casbin policy

Revision ID: b2c4e6f8a1d3
Revises: af98731bb4eb
Create Date: 2026-09-29 16:00:00

"""

from typing import Sequence, Union

import fastapi_users_db_sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "b2c4e6f8a1d3"
down_revision: Union[str, Sequence[str], None] = "af98731bb4eb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

GUID = fastapi_users_db_sqlalchemy.generics.GUID

# Role -> (object, action) permissions. Users are mapped to roles with "g" rules.
POLICY = [
    ("admin", "users", "read"),
    ("admin", "users", "write"),
    ("admin", "audit", "read"),
    ("admin", "batches", "read"),
    ("admin", "predictions", "read"),
    ("reviewer", "batches", "read"),
    ("reviewer", "predictions", "read"),
    ("reviewer", "predictions", "relabel"),
    ("auditor", "batches", "read"),
    ("auditor", "predictions", "read"),
    ("auditor", "audit", "read"),
]


def upgrade() -> None:
    op.add_column(
        "user",
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "invitation",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("invited_by", GUID(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_invitation_email", "invitation", ["email"], unique=True)

    batch_status = sa.Enum("received", "processing", "done", "failed", name="batch_status")
    document_status = sa.Enum("queued", "classified", "failed", name="document_status")

    op.create_table(
        "batch",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("status", batch_status, nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "document",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "batch_id", GUID(), sa.ForeignKey("batch.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("blob_key", sa.String(1024), nullable=False),
        sa.Column("status", document_status, nullable=False),
        sa.Column("error", sa.String(1024), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_document_batch_id", "document", ["batch_id"])

    op.create_table(
        "prediction",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "document_id",
            GUID(),
            sa.ForeignKey("document.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("top5", sa.JSON(), nullable=False),
        sa.Column("overlay_key", sa.String(1024), nullable=True),
        sa.Column("model_sha256", sa.String(64), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("relabeled_to", sa.String(64), nullable=True),
        sa.Column("relabeled_by", GUID(), sa.ForeignKey("user.id"), nullable=True),
        sa.Column("relabeled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_prediction_created_at", "prediction", ["created_at"])

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("actor_id", GUID(), sa.ForeignKey("user.id"), nullable=True),
        sa.Column("actor", sa.String(320), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("target", sa.String(256), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])

    casbin_rule = op.create_table(
        "casbin_rule",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ptype", sa.String(8), nullable=False),
        sa.Column("v0", sa.String(255), nullable=False),
        sa.Column("v1", sa.String(255), nullable=False),
        sa.Column("v2", sa.String(255), nullable=True),
    )
    op.bulk_insert(casbin_rule, [{"ptype": "p", "v0": r, "v1": o, "v2": a} for r, o, a in POLICY])


def downgrade() -> None:
    op.drop_table("casbin_rule")
    op.drop_index("ix_audit_log_created_at", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_prediction_created_at", table_name="prediction")
    op.drop_table("prediction")
    op.drop_index("ix_document_batch_id", table_name="document")
    op.drop_table("document")
    op.drop_table("batch")
    op.drop_index("ix_invitation_email", table_name="invitation")
    op.drop_table("invitation")
    op.drop_column("user", "created_at")
    sa.Enum(name="document_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="batch_status").drop(op.get_bind(), checkfirst=True)
