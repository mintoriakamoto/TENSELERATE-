"""
Memory guardian for the unlocked CMP 170HX - retaining 80 GB *safely*.

No software can make marginal HBM cells reliable; 80 GB stability is a property
of the individual card's silicon. What the engine CAN guarantee is that
instability never reaches a served session. The guardian holds the card at its
unlocked capacity but budgets only memory that has been PROVEN good, and
degrades capacity before it ever degrades an answer.

Three layers, matching what the hardware and the unlock actually allow:

  1. **Persistence.** The unlock is volatile; the cmpunlocker systemd daemon
     rewrites the capacity registers after every reset. Row-remapping (below)
     persists in hardware once ECC is on.
  2. **ECC + row-remapping.** GA100 is Ampere: it has 512 spare DRAM rows and a
     hardware row-remapper that retires degrading cells and persists across
     resets - BUT only when ECC is enabled to detect the errors that trigger
     it. ECC is on the unlock's unresolved list, so we treat "ECC on" as the
     single biggest stability win when the card grants it, and fall back to a
     software carve-out when it does not.
  3. **Soak + carve-out.** Test the whole claimed range; the unstable tail is
     quarantined and the engine budgets `usable = verified - quarantined`. A
     card that cannot hold 80 GB degrades to the largest verified tier
     (80 -> 64 -> 40), and the 1M-ctx / 400-tok/s requirement is met at every
     one of those - so retention never costs the product.

This module is the health model and the safe-capacity policy: CPU-testable, it
takes readings (nvidia-smi capacity, ECC state, Xid/ECC counts, soak result)
and decides what the planner and scheduler may budget. Live GPU calls belong to
`doctor`; the policy that consumes them lives here so it can be tested.
"""
from __future__ import annotations

from dataclasses import dataclass

GiB = 1024 ** 3

# the stable capacity tiers a CMP can land on, largest first; the guardian
# degrades to the largest tier the card actually verifies
STABLE_TIERS_GIB = (80.0, 64.0, 40.0)
# minimum verified capacity to serve the model at all (weights + one window)
MIN_SERVE_GIB = 40.0


class MemoryUnsafe(RuntimeError):
    """Raised when no verified-stable capacity can serve the model."""


@dataclass
class MemoryHealth:
    """
    A snapshot of one CMP's memory trustworthiness.

    reported_gib   : what nvidia-smi reports after the unlock (e.g. 80)
    verified_gib   : capacity a soak test has actually proven stable
    ecc_enabled    : ECC on -> the hardware row-remapper can retire bad cells
    quarantined_gib: regions carved out as bad (soak or live-detected)
    xid_errors     : fatal GPU errors seen since reset (any > 0 is disqualifying)
    uncorrectable_ecc : double-bit ECC errors (data corruption already happened)
    """
    reported_gib: float
    verified_gib: float
    ecc_enabled: bool = False
    quarantined_gib: float = 0.0
    xid_errors: int = 0
    uncorrectable_ecc: int = 0
    remapped_rows: int = 0
    remap_rows_available: int = 512      # GA100 spare rows

    @property
    def usable_gib(self) -> float:
        """Capacity the engine may budget: proven good, minus carve-outs."""
        return max(0.0, min(self.verified_gib, self.reported_gib)
                   - self.quarantined_gib)

    @property
    def remap_exhausted(self) -> bool:
        """No spare rows left - the card can no longer retire new bad cells."""
        return self.remapped_rows >= self.remap_rows_available

    @property
    def serve_ready(self) -> bool:
        """
        Safe to serve only if: enough verified capacity, no fatal errors, no
        corruption that already slipped through, and the hardware can still
        retire a new bad cell (or ECC is on to catch it).
        """
        return (self.usable_gib >= MIN_SERVE_GIB
                and self.xid_errors == 0
                and self.uncorrectable_ecc == 0
                and not self.remap_exhausted)

    def safe_tier_gib(self) -> float:
        """
        The largest stable tier this card's usable capacity supports. Serving
        at a tier the card verifies - rather than the raw reported number - is
        how 80 GB is 'retained': held when proven, stepped down when not.
        """
        u = self.usable_gib
        for tier in STABLE_TIERS_GIB:
            if u >= tier:
                return tier
        raise MemoryUnsafe(
            f"usable {u:.1f} GiB below the {MIN_SERVE_GIB:.0f} GiB serve floor; "
            f"this card cannot safely serve at any tier")

    def reason(self) -> str:
        """One line on why the card is or is not serve-ready."""
        if self.xid_errors:
            return f"{self.xid_errors} Xid error(s) since reset - not serve-ready"
        if self.uncorrectable_ecc:
            return (f"{self.uncorrectable_ecc} uncorrectable ECC error(s) - data "
                    "corruption already occurred, not serve-ready")
        if self.remap_exhausted:
            return "row-remap pool exhausted - card can retire no more bad cells"
        if self.usable_gib < MIN_SERVE_GIB:
            return f"usable {self.usable_gib:.1f} GiB below serve floor"
        ecc = "ECC on (row-remap active)" if self.ecc_enabled else \
            "ECC OFF - relying on soak carve-out, no live bad-cell retirement"
        return (f"serve-ready at {self.safe_tier_gib():.0f} GiB tier "
                f"(usable {self.usable_gib:.1f} GiB, {ecc})")


@dataclass
class SoakPlan:
    """
    How to prove a capacity stable before trusting it. A card is verified only
    by exercising the WHOLE claimed range under load - a capacity screenshot is
    not proof.
    """
    fraction_to_exercise: float = 0.95   # cover ~all of VRAM, not a corner
    hours: float = 6.0                   # long enough to surface refresh drift
    max_correctable_ecc: int = 0         # any SBE at the top of range is a flag

    def passes(self, health: MemoryHealth, correctable_ecc: int) -> bool:
        """A soak passes only with zero fatal errors and SBE at/below budget."""
        return (health.xid_errors == 0
                and health.uncorrectable_ecc == 0
                and correctable_ecc <= self.max_correctable_ecc)


def choose_machine_profile(health: MemoryHealth) -> str:
    """
    Map a card's verified health to the `MACHINE_HW` profile the planner should
    use, so `plan` budgets proven-good memory. Retention in one call: an 80 GB
    card that soaks clean plans as `cmp170hx`; one that only holds 64/40 plans
    as the smaller profile - still meeting the requirement, never overcommitting
    memory the card cannot hold.
    """
    tier = health.safe_tier_gib()        # raises MemoryUnsafe if unservable
    return {80.0: "cmp170hx", 64.0: "cmp170hx-64",
            40.0: "cmp170hx-40"}[tier]
