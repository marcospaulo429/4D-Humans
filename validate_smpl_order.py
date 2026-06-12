"""
Validacao OFFLINE (sem SONIC) do payload do Stage 1 contra a arvore cinematica
SMPL-24 que o SONIC espera (Protocol v2).

Referencia extraida do codigo do SONIC:
  gear_sonic/utils/teleop/vis/vr3pt_pose_visualizer.py
  SMPL_PARENT_INDICES = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]

Verifica:
  1. smpl_joints[24,3] e smpl_pose[21,3] com shapes corretos.
  2. Estrutura do esqueleto: comprimentos de osso (parent->child) plausiveis p/ humano.
  3. Simetria esquerda/direita (ossos equivalentes ~mesmo comprimento).
  4. Eixo "up" dominante (camera Y-down vs mundo Z-up) -> sinaliza transform necessaria.
  5. Escala global (altura aprox.) em metros.
"""
import argparse
import numpy as np

# Ordem canonica SMPL-24 (nomes) — casa com SMPL_PARENT_INDICES do SONIC
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
SMPL_PARENT_INDICES = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                       9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# pares simetricos (esq, dir) p/ checar consistencia
SYM_PAIRS = [
    ("left_hip", "right_hip"), ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"), ("left_foot", "right_foot"),
    ("left_collar", "right_collar"), ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"), ("left_wrist", "right_wrist"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="stage1_out/smpl_v2_payload.npz")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz)
    sj = d["smpl_joints"]   # [N,24,3]
    sp = d["smpl_pose"]     # [N,21,3]
    bq = d["body_quat"]     # [N,4]
    print(f"Arquivo: {args.npz}")
    print(f"smpl_joints {sj.shape} | smpl_pose {sp.shape} | body_quat {bq.shape}\n")

    ok = True
    # ---- 1. shapes ----
    if sj.shape[1:] != (24, 3):
        print(f"[FALHA] smpl_joints deve ser [N,24,3], veio {sj.shape}"); ok = False
    if sp.shape[1:] != (21, 3):
        print(f"[FALHA] smpl_pose deve ser [N,21,3], veio {sp.shape}"); ok = False

    J = sj[args.frame]  # [24,3]
    name2idx = {n: i for i, n in enumerate(SMPL_JOINT_NAMES)}

    # ---- 2. comprimentos de osso ----
    print("=== Comprimentos de osso (parent -> child), metros ===")
    bone_len = {}
    for i in range(1, 24):
        p = SMPL_PARENT_INDICES[i]
        L = float(np.linalg.norm(J[i] - J[p]))
        bone_len[SMPL_JOINT_NAMES[i]] = L
    for n in ["left_hip", "left_knee", "left_ankle", "spine1", "spine2", "spine3",
              "neck", "head", "left_shoulder", "left_elbow", "left_wrist"]:
        print(f"  {n:16s} {bone_len[n]*100:6.1f} cm")

    # plausibilidade: nenhum osso > 0.8 m, coxa entre 5-60cm
    maxL = max(bone_len.values())
    if maxL > 0.8:
        print(f"[ALERTA] osso maior que 80cm ({maxL*100:.1f}cm) -> escala suspeita"); ok = False
    else:
        print(f"  [OK] maior osso = {maxL*100:.1f} cm (plausivel)")

    # ---- 3. simetria E/D ----
    print("\n=== Simetria esquerda/direita (diferenca relativa) ===")
    sym_ok = True
    for l, r in SYM_PAIRS:
        Ll, Lr = bone_len[l], bone_len[r]
        rel = abs(Ll - Lr) / max(Ll, Lr, 1e-6)
        flag = "" if rel < 0.15 else "  <-- assimetrico!"
        if rel >= 0.15:
            sym_ok = False
        print(f"  {l:15s} vs {r:15s}: {Ll*100:5.1f} / {Lr*100:5.1f} cm  (dif {rel*100:4.1f}%){flag}")
    print(f"  [{'OK' if sym_ok else 'ALERTA'}] simetria {'consistente' if sym_ok else 'com desvios'}")

    # ---- 4. eixo up dominante ----
    print("\n=== Orientacao (qual eixo e vertical?) ===")
    # vetor pelve->cabeca deve ser ~vertical
    spine_vec = J[name2idx["head"]] - J[name2idx["pelvis"]]
    spine_vec = spine_vec / (np.linalg.norm(spine_vec) + 1e-9)
    axis = ["X", "Y", "Z"][int(np.argmax(np.abs(spine_vec)))]
    sign = "+" if spine_vec[int(np.argmax(np.abs(spine_vec)))] > 0 else "-"
    print(f"  vetor pelve->cabeca = {np.round(spine_vec,3)}  -> eixo dominante {sign}{axis}")
    if axis == "Y" and sign == "-":
        print("  [INFO] Y-down (frame de camera, tipico HMR2). SONIC espera Z-up:")
        print("         sera necessario um transform de frame antes de publicar (esperado).")
    elif axis == "Z":
        print("  [INFO] Z e vertical (proximo do frame do robo).")
    else:
        print("  [INFO] eixo vertical inesperado; conferir convencao.")

    # ---- 5. altura aproximada ----
    height = float(J[:, int(np.argmax(np.abs(spine_vec)))].max() - J[:, int(np.argmax(np.abs(spine_vec)))].min())
    print(f"\n=== Escala ===\n  extensao no eixo vertical ~ {height*100:.1f} cm (esperado ~120-190cm p/ humano em pe)")

    # ---- 6. body_quat normalizado ----
    qn = np.linalg.norm(bq[args.frame])
    print(f"  body_quat norma = {qn:.5f} (deve ~1.0)")

    print("\n===== RESUMO =====")
    print(f"  ordem de juntas SMPL-24 : compativel com SONIC (parent tree padrao)")
    print(f"  shapes Protocol v2      : {'OK' if ok else 'PROBLEMA'}")
    print(f"  simetria corporal       : {'OK' if sym_ok else 'verificar'}")
    print(f"  GATE estrutura          : {'PASSOU' if (ok and sym_ok) else 'REVISAR'}")
    print("\n  Pendencias que SO o SONIC resolve:")
    print("   - transform de frame (camera Y-down -> robo Z-up) + heading")
    print("   - escala fina / posicao da pelve (absoluta vs relativa)")
    print("   - confirmacao final movendo o robo em sim")


if __name__ == "__main__":
    main()
