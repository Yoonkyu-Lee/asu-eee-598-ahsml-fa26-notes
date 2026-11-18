"""Read CPU package energy (Intel RAPL via Windows "Energy Meter" counters) and find P-cores vs E-cores.

Windows exposes RAPL through the PDH counter set "Energy Meter" without admin rights.
`Energy` is a cumulative counter; we read it before and after a workload and divide by time.
Its unit is calibrated once against the `Power` counter (mW) by `calibrate()`.
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

pdh = ctypes.WinDLL("pdh")
PDH_FMT_LARGE = 0x00000400
PDH_FMT_DOUBLE = 0x00000200
PWH = 3.6e-9  # the Energy counter counts picowatt-hours (calibrated 2026-10-30: 3.5998e-9 J/unit)


class _FmtValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("pad", wintypes.DWORD), ("value", ctypes.c_longlong)]


class _FmtValueD(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("pad", wintypes.DWORD), ("value", ctypes.c_double)]


class _Counter:
    def __init__(self, path: str):
        self.q = wintypes.HANDLE()
        self.c = wintypes.HANDLE()
        if pdh.PdhOpenQueryW(None, None, ctypes.byref(self.q)):
            raise OSError("PdhOpenQuery failed")
        if pdh.PdhAddEnglishCounterW(self.q, path, None, ctypes.byref(self.c)):
            raise OSError(f"PdhAddEnglishCounter failed for {path}")

    def large(self) -> int:
        pdh.PdhCollectQueryData(self.q)
        v = _FmtValue()
        pdh.PdhGetFormattedCounterValue(self.c, PDH_FMT_LARGE, None, ctypes.byref(v))
        return v.value

    def double(self) -> float:
        pdh.PdhCollectQueryData(self.q)
        v = _FmtValueD()
        pdh.PdhGetFormattedCounterValue(self.c, PDH_FMT_DOUBLE, None, ctypes.byref(v))
        return v.value


class PackageEnergy:
    """meter = PackageEnergy(); meter.start(); ...work...; joules, seconds = meter.stop()"""

    def __init__(self, domain: str = "RAPL_Package0_PKG", joules_per_unit: float | None = None):
        self.energy = _Counter(rf"\Energy Meter({domain})\Energy")
        self.jpu = joules_per_unit if joules_per_unit is not None else PWH

    def start(self) -> None:
        self.t0, self.e0 = time.perf_counter(), self.energy.large()

    def stop(self) -> tuple[float, float]:
        e1, t1 = self.energy.large(), time.perf_counter()
        if self.jpu is None:
            raise RuntimeError("call calibrate() first or pass joules_per_unit")
        return (e1 - self.e0) * self.jpu, t1 - self.t0


def calibrate(seconds: float = 5.0, domain: str = "RAPL_Package0_PKG") -> dict:
    """Compare the cumulative Energy counter against the averaged Power counter (mW) over an idle window."""
    e = _Counter(rf"\Energy Meter({domain})\Energy")
    p = _Counter(rf"\Energy Meter({domain})\Power")
    p.double()
    e0, t0, powers = e.large(), time.perf_counter(), []
    while time.perf_counter() - t0 < seconds:
        time.sleep(0.25)
        powers.append(p.double())
    e1, t1 = e.large(), time.perf_counter()
    avg_mw = sum(powers) / len(powers)
    units_per_s = (e1 - e0) / (t1 - t0)
    return {"avg_power_W": avg_mw / 1000, "energy_units_per_s": units_per_s,
            "joules_per_unit": (avg_mw / 1000) / units_per_s if units_per_s else None}


# ---------------------------------------------------------------- P-cores vs E-cores

class _CpuSet(ctypes.Structure):
    _fields_ = [("Size", wintypes.DWORD), ("Type", wintypes.DWORD), ("Id", wintypes.DWORD),
                ("Group", wintypes.WORD), ("LogicalProcessorIndex", ctypes.c_ubyte), ("CoreIndex", ctypes.c_ubyte),
                ("LastLevelCacheIndex", ctypes.c_ubyte), ("NumaNodeIndex", ctypes.c_ubyte),
                ("EfficiencyClass", ctypes.c_ubyte), ("AllFlags", ctypes.c_ubyte),
                ("Reserved", wintypes.DWORD), ("AllocationTag", ctypes.c_ulonglong)]  # 32 bytes, matches SYSTEM_CPU_SET_INFORMATION


def core_classes() -> dict[int, list[int]]:
    """EfficiencyClass -> logical CPU indices. Higher class = faster cores (P-cores on hybrid Intel)."""
    k32 = ctypes.WinDLL("kernel32")
    need = wintypes.ULONG()
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
    buf = ctypes.create_string_buffer(need.value)
    if not k32.GetSystemCpuSetInformation(buf, need, ctypes.byref(need), None, 0):
        raise OSError("GetSystemCpuSetInformation failed")
    out: dict[int, list[int]] = {}
    off = 0
    while off < need.value:
        cs = _CpuSet.from_buffer_copy(buf.raw[off: off + ctypes.sizeof(_CpuSet)])
        out.setdefault(cs.EfficiencyClass, []).append(cs.LogicalProcessorIndex)
        off += cs.Size
    return {k: sorted(v) for k, v in sorted(out.items())}


if __name__ == "__main__":
    print("core classes:", core_classes())
    print("calibration:", calibrate(3.0))
