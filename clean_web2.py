"""Labels from the manual review of both targeted rounds. Output: data/web_clean2. Not merged, not trained on."""
import collections
import csv
import os
import shutil

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = [os.path.join(HERE, "data", f) for f in ("web_raw_110", "web_raw_real")]
OUT = os.path.join(HERE, "data", "web_clean2")

KEEP = {
    "110": """-uZPS7Ru-uI_00224 -uZPS7Ru-uI_01120 20Do_Cse2AI_00357 4xhi2-SeJbc_00641 4xhi2-SeJbc_03207 6MiLwdmbMC0_06400
      8urkOzanqs8_00237 8urkOzanqs8_01187 9T-sKU4OuXI_00763 9T-sKU4OuXI_01909 AGKIfP_MvkM_00415 AhBBvXaGnTQ_00610 AhBBvXaGnTQ_04880_p1
      Blw8YC4Wkzk_00126 LWPS5T1DTak_00007 MAa-wBs3cis_00013 Rc5A-k4SQFc_00208 aoMWPZYGhQ8_00532 aoMWPZYGhQ8_04263
      b-sVdigCzOM_00294 b-sVdigCzOM_01470 btEie7v0bqA_00166 cZ_KteOC-zM_00027 dMQKTjJnvSc_00092 fPMzOsOfEc0_00091 kw_s0K_zbqc_00797
      l5NP4IQeqc0_00103 ltOiEAUqAi8_00546 lyFxEpLI-sM_02888 oQqtIaQkUmQ_00472
      p9qcKpIwQuY_00130 pHfEFzh8DaY_00415 qVIzSMux720_00486 tl2ug5m_jcg_00278
      txpM9oikSJU_00160 uAnyqEcPss0_00653 vK4hItL7_vc_00800 vsyOddNpbYw_00220 xCYF_whNa9E_00399 yObSCdnoIaw_03136 yXbfCIdK6-c_00675
      O2BQe8UDNB8_00088 O2BQe8UDNB8_00704_p1 Yd5DOR69F-8_00009 Yd5DOR69F-8_00048""",
    "111": """20Do_Cse2AI_01252 4bF6UzuKPvs_00032 4bF6UzuKPvs_00194 75whjeTINRc_00026 75whjeTINRc_00105 FTb5WyIIooI_00047 FTb5WyIIooI_00383
      JDjL_QAe3JY_04780 JQzG3H2xahU_00078 Kx26RbRX_R8_00038_p1 M08JJDiTg7c_00800 TCbP8kYuhxo_00098 TCbP8kYuhxo_00296 TjhX4L54pUM_00098 TjhX4L54pUM_00491_p1
      WHBTxrh0dhw_00270 c2xjKH8aHFc_00006 hDObycqbCbY_00024 h0IRNVZK1qk_00246 kw_s0K_zbqc_03985 mCHP6y7uvRk_00623 pNJXedpweSM_00528 vBNQjJFhqWo_00583
      vK4hItL7_vc_03200 yFuzsPiIRu4_00503 r0uT8AM0z4Y_00603 Rc5A-k4SQFc_01040_p1 Pgi5QGHjXoo_00505 Pgi5QGHjXoo_02021 Pgi5QGHjXoo_04043_p1
      QLPgeFJtHkY_00098 QLPgeFJtHkY_00791 MOq6vA6Sr6I_00124 In4erODZ34s_00156 JjPoWSnyvMM_00595 KIIbF6BM6nY_00091 KIIbF6BM6nY_00459 _mhmEtHg33U_00158_p1
      kd4EQR3XHws_06297 qTeWySaKL9A_00395 qTeWySaKL9A_00632_p2 sbbpVsIkEx4_00031 4sNdU1npKAo_00037 WP0TLge52jE_00302 3A4yiYsYZ5A_00314 GF6Lo9whqcM_00186""",
    "100": """4RO26u1S-mQ_00428 7gEHzfd5WSg_03886 FdfgwADHI7k_00800 FdfgwADHI7k_06400 I1GIH2TVotI_00010 JDjL_QAe3JY_02390 ePzMu4PnKH0_00148
      qRqdL4OLmsU_00800 qRqdL4OLmsU_06400 cOw-i3cIILM_00800 cOw-i3cIILM_04000 8xrpnxIfLXI_00007 zK5WXahMYDA_00017 zPXcye1IdfM_02628 uuHt4xomAnM_00585
      7wFAq9ECJDE_00901 7wFAq9ECJDE_01802_p1 2R3LF39gCjU_00623_p1 LoUciuL0hdY_00011 Oy3Xxev-FtE_00861 Oy3Xxev-FtE_01206 YhdKx9eq7hM_00015 ZY2cgvoPBMc_00212 ZY2cgvoPBMc_01701
      nrGv2o64M3I_00661 rdNjGBW4oZ8_00193 rdNjGBW4oZ8_00968 txjcDC5CHL8_00418 txjcDC5CHL8_01255_p1 txjcDC5CHL8_01673_p3
      JjPoWSnyvMM_02381_p2 l82MizdVw4I_01538_p1 GF6Lo9whqcM_00746 ApMmK_TiomA_00323""",
    "000": """0LocWUPyLAo_00091 1hmHgnCicpw_00549 2R3LF39gCjU_00207 3pRtgQcVAzc_01746_p1 4nR4322yqF4_02377 7gEHzfd5WSg_00777 8dIsY-MIdvg_00105_p1 9a0BvgsetzA_00603
      B28F9wifRzQ_00196 FIIrcyINy9I_00371 I5BV1Yq1vro_00707 I5BV1Yq1vro_00943 K56jam7gWkI_01684 ODsKDTPnmTY_00800 P5nNW0GoMy8_00600
      TkQiEvS9hbo_00019 TkQiEvS9hbo_00097 VFbIq_lag8M_02144 cGz01nrf0ao_00861 dTq95KFdFKg_00542 eOn_AiP58k0_00124 iEMNXXVqx7k_00800 iWpFhuh2bDU_00550
      j4zm63ic4Pg_01444_p1 jcgt_MyGejQ_00800 pTvecu7WzFk_00271 qTeWySaKL9A_00079 siZwd5nQ87c_00205 vrNCweZIZEQ_00311 zUXZZBzePRM_00032
      zqK5PWICaTQ_01835_p1 Z0wDW2T2R2c_00616 Kx26RbRX_R8_00019""",
}


def norm(s):
    return s.replace("I", "l").replace("O", "0").lower()


index = collections.defaultdict(list)
for folder in FOLDERS:
    m = {r["crop"]: r for r in csv.DictReader(open(os.path.join(folder, "manifest.csv"), encoding="utf-8"))}
    for f in os.listdir(os.path.join(folder, "crops")):
        index[norm(f[:-4])].append((f, folder, m[f]))

items, problems = [], []
for label, text in KEEP.items():
    for n in text.split():
        n = n if "_p" in n[-3:] else n + "_p0"
        hits = index.get(norm(n), [])
        if len(hits) == 1:
            items.append((*hits[0], label))
        else:
            problems.append((n, len(hits)))
print("unresolved", problems)
assert len({i[0] for i in items}) == len(items), "duplicate crop"

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(os.path.join(OUT, "images"))
with open(os.path.join(OUT, "labels.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["filename", "thobe", "ghutra", "igal", "video_id", "url", "second"])
    for n, folder, m, label in sorted(items, key=lambda x: x[0]):
        shutil.copy2(os.path.join(folder, "crops", n), os.path.join(OUT, "images", n))
        w.writerow([n, *label, m["video_id"], m["url"], m["second"]])

C, W, H = 8, 150, 190
for label in KEEP:
    ns = sorted(i[0] for i in items if i[3] == label)
    rows = (len(ns) + C - 1) // C
    S = Image.new("RGB", (C * W, rows * (H + 14)), "white")
    D = ImageDraw.Draw(S)
    for i, n in enumerate(ns):
        im = Image.open(os.path.join(OUT, "images", n)).convert("RGB")
        im.thumbnail((W - 4, H))
        x, y = i % C * W, i // C * (H + 14)
        S.paste(im, (x, y + 14))
        D.text((x + 2, y + 1), n[:-4][:24], fill="black")
    S.save(os.path.join(OUT, f"review_{label}.jpg"), quality=90)

print("kept", len(items))
print("labels", sorted(collections.Counter(i[3] for i in items).items()))
print("videos", {l: len({i[2]["video_id"] for i in items if i[3] == l}) for l in KEEP})
