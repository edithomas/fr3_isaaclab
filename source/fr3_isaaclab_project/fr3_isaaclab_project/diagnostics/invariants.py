"""FR3 USD asset invariants — pure Python, no Isaac Sim imports.

These functions encode the structural invariants the experimental tooling relies
on for the Franka Research 3 arm under Isaac Lab v2.1.0 / Isaac Sim 4.5:

* The end-effector body is ``fr3_hand`` (verified against the FR3 USD; the
  ``fr3_reach`` env config points ``SceneEntityCfg.body_names`` at it).
* The seven revolute arm joints are ``fr3_joint1`` … ``fr3_joint7``
  (``fr3_joint8`` and ``fr3_hand_joint`` are fixed joints, not DOFs).
* The articulation exposes a nonzero set of applied rigid bodies.

The functions are pure (take plain lists/ints, return dataclasses/bools) so they
can be unit-tested on macOS without a GPU. The headless ``check_fr3_asset.py``
script feeds them the joint/body names it scrapes from the live USD stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Expected FR3 asset structure (verified against ISAAC_NUCLEUS_DIR/.../fr3.usd)
# ---------------------------------------------------------------------------
EXPECTED_EE_BODY: str = "fr3_hand"
"""End-effector body name used by ``fr3_reach`` rewards/commands."""

EXPECTED_ARM_JOINTS: tuple[str, ...] = (
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
)
"""The seven revolute arm DOFs. ``fr3_joint8``/``fr3_hand_joint`` are fixed."""

MIN_RIGID_BODIES: int = 1
"""A real articulation must expose at least one applied rigid body."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class AssetReport:
    """Structured result of classifying an FR3 USD asset.

    ``ok`` is the single boolean the asset-check script exits on. ``violations``
    is the ordered list of human-readable invariant failures; an empty list with
    ``ok=True`` means every invariant held. Keeping this a dataclass (rather than
    raising) lets the script print a full diagnosis before returning a nonzero
    exit code, which is what makes swallowed-error regressions detectable.
    """

    ok: bool = True
    violations: list[str] = field(default_factory=list)
    missing_arm_joints: list[str] = field(default_factory=list)
    extra_arm_joints: list[str] = field(default_factory=list)
    ee_body_found: bool = False
    num_rigid_bodies: int = 0
    arm_joint_names: list[str] = field(default_factory=list)
    gripper_joint_names: list[str] = field(default_factory=list)
    ee_candidates: list[str] = field(default_factory=list)

    def add_violation(self, msg: str) -> None:
        self.violations.append(msg)
        self.ok = False


# ---------------------------------------------------------------------------
# Individual invariant checks
# ---------------------------------------------------------------------------
def check_arm_joints_present(joint_names: list[str]) -> tuple[list[str], list[str]]:
    """Return ``(missing, extra)`` arm joints vs. the expected seven.

    ``extra`` are arm joints present that match ``fr3_joint*`` but are not in the
    expected set (e.g. a stray ``fr3_joint8`` if a USD variant promoted it to a
    revolute DOF — it would then be an unexpected DOF, not a fixed joint).
    """
    names = set(joint_names)
    missing = [j for j in EXPECTED_ARM_JOINTS if j not in names]
    # arm joints = those matching the fr3_joint<digit> revolute pattern
    present_arm = [j for j in joint_names if _is_arm_revolute_joint(j)]
    extra = [j for j in present_arm if j not in EXPECTED_ARM_JOINTS]
    return missing, extra


def _is_arm_revolute_joint(name: str) -> bool:
    """True if ``name`` looks like ``fr3_joint<N>`` for a single digit N."""
    if not name.startswith("fr3_joint"):
        return False
    suffix = name[len("fr3_joint"):]
    return suffix.isdigit() and len(suffix) == 1


def check_ee_body_present(body_names: list[str]) -> bool:
    """True if ``fr3_hand`` is among the rigid bodies."""
    return EXPECTED_EE_BODY in body_names


def check_nonzero_rigid_bodies(body_names: list[str]) -> bool:
    """True if the stage exposes at least ``MIN_RIGID_BODIES`` rigid body."""
    return len(body_names) >= MIN_RIGID_BODIES


def classify_asset(
    joint_names: list[str],
    body_names: list[str],
    articulation_roots: list[str] | None = None,
) -> AssetReport:
    """Run every FR3 asset invariant and return a structured report.

    Args:
        joint_names: All joint prim names scraped from the stage (any order).
        body_names: All applied-rigid-body names scraped from the stage.
        articulation_roots: Optional list of articulation-root prim paths. An
            empty list is *not* itself a violation (the FR3 USD may mark the root
            on a parent prim that the traversal reports differently), but it is
            recorded for diagnosis.

    The caller decides the nonzero exit code from ``report.ok``.
    """
    report = AssetReport()

    # arm joints
    missing, extra = check_arm_joints_present(joint_names)
    report.missing_arm_joints = missing
    report.extra_arm_joints = extra
    if missing:
        report.add_violation(
            f"Missing expected arm joints: {missing} (expected {list(EXPECTED_ARM_JOINTS)})"
        )
    if extra:
        report.add_violation(
            f"Unexpected extra revolute arm joints: {extra} (only {list(EXPECTED_ARM_JOINTS)} are DOFs)"
        )
    report.arm_joint_names = [j for j in joint_names if _is_arm_revolute_joint(j)]

    # gripper joints (informational only — not a DOF invariant)
    report.gripper_joint_names = [
        j for j in joint_names if "finger" in j.lower() or "gripper" in j.lower()
    ]

    # end-effector body
    report.ee_body_found = check_ee_body_present(body_names)
    if not report.ee_body_found:
        report.add_violation(
            f"End-effector body '{EXPECTED_EE_BODY}' not found among rigid bodies"
        )

    # nonzero rigid bodies
    report.num_rigid_bodies = len(body_names)
    if not check_nonzero_rigid_bodies(body_names):
        report.add_violation(
            f"Expected >= {MIN_RIGID_BODIES} rigid body, found {len(body_names)}"
        )

    # informational: end-effector candidates (keyword match) for manual review
    ee_keywords = (
        "hand", "tcp", "link8", "link_8", "ee", "end_effector",
        "flange", "tool0", "gripper", "gripper_center",
    )
    report.ee_candidates = [
        b for b in body_names if any(kw in b.lower() for kw in ee_keywords)
    ]

    # informational only: articulation roots
    if articulation_roots is not None and not articulation_roots:
        # not a hard invariant — recorded for diagnosis only
        report.violations.append(
            "Note: no ArticulationRootAPI prims traversed (not an invariant failure)"
        )

    return report
