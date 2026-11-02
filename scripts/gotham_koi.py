"""Patch git-koi: the black body becomes black and gold, and the koi always cruises.

Usage: python gotham_koi.py path/to/git-koi
"""
import re, sys
root = sys.argv[1]
p = root + "/src/coat.rs"
s = open(p).read()
def sub(old, new):
    global s
    assert old in s, old
    s = s.replace(old, new, 1)
sub("            Self::White | Self::Black => &WHITE_RAMP,\n            Self::Asagi => &ORANGE_RAMP,",
    "            Self::White => &WHITE_RAMP,\n            Self::Black => &GOTHAM_GOLD_RAMP,\n            Self::Asagi => &ORANGE_RAMP,")
sub("            Self::Asagi | Self::Gold => &RED_RAMP,",
    "            Self::Black => &PLATINUM_RAMP,\n            Self::Asagi | Self::Gold => &RED_RAMP,")
sub("            Self::White | Self::Black | Self::Tea | Self::Platinum => &STREAK_GOLD_RAMP,",
    "            Self::White | Self::Tea | Self::Platinum => &STREAK_GOLD_RAMP,")
sub("        match (self, pigment) {\n",
    "        match (self, pigment) {\n            (Self::Black, _) => &GOTHAM_GOLD_RAMP,\n")
sub("const BLACK_RAMP: [[u8; 4]; 3] = [[24, 20, 37, 255], [38, 43, 68, 255], [58, 68, 102, 255]];",
    "const BLACK_RAMP: [[u8; 4]; 3] = [[5, 5, 7, 255], [14, 14, 17, 255], [28, 27, 31, 255]];\n"
    "const GOTHAM_GOLD_RAMP: [[u8; 4]; 4] = [[122, 84, 18, 255], [184, 134, 11, 255], [222, 170, 40, 255], [255, 218, 92, 255]];")
s = s.replace("(Self::Black, Pigment::Beni | Pigment::Orenji) => &COPPER_RAMP,\n            (Self::Black, _) => &GOLD_RAMP,",
              "(Self::Black, _) => &GOTHAM_GOLD_RAMP,")
s = s.replace("            Self::Black => &GOLD_RAMP,\n            Self::Asagi => &ORANGE_RAMP,",
              "            Self::Black => &GOTHAM_GOLD_RAMP,\n            Self::Asagi => &ORANGE_RAMP,")
sub("        if dither_threshold <= calculate_pigment_development(pigment, self.color_fraction) {",
    "        if true || dither_threshold <= calculate_pigment_development(pigment, self.color_fraction) {")
open(p, "w").write(s)
p = root + "/src/behavior.rs"
s = open(p).read()
s, n = re.subn(r"(pub fn select_behavior\(koi_development: &KoiDevelopment\) -> Behavior \{)",
               r"\1\n    let _ = koi_development;\n    return Behavior::Cruise;\n    #[allow(unreachable_code)]", s)
assert n == 1
open(p, "w").write(s)
