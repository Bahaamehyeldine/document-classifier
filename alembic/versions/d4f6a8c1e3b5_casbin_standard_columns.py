"""casbin_rule: add the standard v3..v5 columns casbin-sqlalchemy-adapter expects

Revision ID: d4f6a8c1e3b5
Revises: b2c4e6f8a1d3
Create Date: 2026-10-15
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4f6a8c1e3b5"
down_revision: Union[str, Sequence[str], None] = "b2c4e6f8a1d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EXTRA_COLUMNS = ("v3", "v4", "v5")


def upgrade() -> None:
    for name in EXTRA_COLUMNS:
        op.add_column("casbin_rule", sa.Column(name, sa.String(255), nullable=True))


def downgrade() -> None:
    for name in reversed(EXTRA_COLUMNS):
        op.drop_column("casbin_rule", name)
