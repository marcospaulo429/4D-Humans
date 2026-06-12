"""
Stage 1 — Conversor 4D-Humans (HMR2.0) -> payload SMPL "Protocol v2".

Objetivo (sem SONIC): a partir de uma imagem/pasta, rodar HMR2.0, extrair os
parametros SMPL por frame e converte-los para o formato do Protocol v2:

    smpl_pose   [N, 21, 3]   axis-angle (21 juntas de corpo; descarta as 2 maos)
    smpl_joints [N, 24, 3]   posicoes 3D das 24 juntas SMPL
    body_quat   [N, 4]       quaternion da raiz, ordem (w, x, y, z)
    frame_index [N]          contador monotonico

O HMR2.0 entrega 'body_pose' como matrizes de rotacao [23,3,3] e 'global_orient'
[1,3,3]. Aqui validamos a conversao rotmat->axis-angle por *roundtrip* numerico
(axis-angle -> rotmat de volta) e gravamos um .npz com o payload v2.

Uso:
    python stage1_smpl_to_v2.py --img_folder example_data/one --out stage1_out
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import DEFAULT_CHECKPOINT, load_hmr2
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset


def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """[..., 3, 3] -> [..., 3] (axis-angle). Usa scipy (convencao padrao)."""
    flat = rotmat.reshape(-1, 3, 3)
    aa = R.from_matrix(flat).as_rotvec()
    return aa.reshape(rotmat.shape[:-2] + (3,))


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """[..., 3] -> [..., 3, 3]. Inverso do acima, para o roundtrip."""
    flat = aa.reshape(-1, 3)
    rm = R.from_matrix(np.eye(3))  # placeholder p/ shape
    rm = R.from_rotvec(flat).as_matrix()
    return rm.reshape(aa.shape[:-1] + (3, 3))


def rotmat_to_quat_wxyz(rotmat: np.ndarray) -> np.ndarray:
    """[3,3] -> [4] em ordem (w, x, y, z) (convencao do Protocol v2)."""
    xyzw = R.from_matrix(rotmat).as_quat()  # scipy: (x, y, z, w)
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: HMR2 -> SMPL Protocol v2")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--img_folder", type=str, default="example_data/one")
    parser.add_argument("--out", type=str, default="stage1_out")
    parser.add_argument("--detector", type=str, default="regnety", choices=["vitdet", "regnety"])
    parser.add_argument("--file_type", nargs="+", default=["*.jpg", "*.png"])
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[stage1] device = {device}")

    model, model_cfg = load_hmr2(args.checkpoint)
    model = model.to(device).eval()

    # Detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.detector == "vitdet":
        from detectron2.config import LazyConfig
        import hmr2
        cfg_path = Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        d2cfg = LazyConfig.load(str(cfg_path))
        d2cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            d2cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(d2cfg)
    else:
        from detectron2 import model_zoo
        d2cfg = model_zoo.get_config("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
        d2cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        d2cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(d2cfg)

    img_paths = sorted([p for ext in args.file_type for p in Path(args.img_folder).glob(ext)])
    print(f"[stage1] {len(img_paths)} imagem(ns) encontrada(s)")

    smpl_pose_all, smpl_joints_all, body_quat_all, frame_idx_all = [], [], [], []
    max_roundtrip_err = 0.0
    frame_index = 0

    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        det = detector(img_cv2)["instances"]
        valid = (det.pred_classes == 0) & (det.scores > 0.5)
        boxes = det.pred_boxes.tensor[valid].cpu().numpy()
        if len(boxes) == 0:
            print(f"[stage1] {img_path.name}: nenhuma pessoa detectada, pulando")
            continue

        # Pessoa unica: usa a deteccao de maior score (primeira apos filtro)
        boxes = boxes[:1]
        ds = ViTDetDataset(model_cfg, img_cv2, boxes)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            # --- parametros SMPL (matrizes de rotacao) ---
            global_orient = out["pred_smpl_params"]["global_orient"][0].cpu().numpy()  # [1,3,3]
            body_pose = out["pred_smpl_params"]["body_pose"][0].cpu().numpy()           # [23,3,3]

            # --- rotmat -> axis-angle ---
            body_aa = rotmat_to_axis_angle(body_pose)        # [23,3]
            smpl_pose = body_aa[:21]                          # [21,3]  (descarta 2 maos)

            # --- roundtrip de validacao: axis-angle -> rotmat ---
            body_rm_back = axis_angle_to_rotmat(body_aa)      # [23,3,3]
            err = np.abs(body_rm_back - body_pose).max()
            max_roundtrip_err = max(max_roundtrip_err, float(err))

            # --- quaternion da raiz (w,x,y,z) ---
            body_quat = rotmat_to_quat_wxyz(global_orient[0])  # [4]

            # --- 24 juntas SMPL nativas (J_regressor sobre os vertices) ---
            verts = out["pred_vertices"][0]                   # [6890,3] tensor
            J_reg = model.smpl.J_regressor.to(verts.device)   # [24,6890]
            smpl_joints = (J_reg @ verts).cpu().numpy()       # [24,3]

            smpl_pose_all.append(smpl_pose.astype(np.float32))
            smpl_joints_all.append(smpl_joints.astype(np.float32))
            body_quat_all.append(body_quat.astype(np.float32))
            frame_idx_all.append(frame_index)
            print(f"[stage1] frame {frame_index} ({img_path.name}): "
                  f"smpl_pose{smpl_pose.shape} smpl_joints{smpl_joints.shape} "
                  f"body_quat(wxyz)={np.round(body_quat,3)} roundtrip_err={err:.2e}")
            frame_index += 1

    if frame_index == 0:
        print("[stage1] ERRO: nenhum frame processado.")
        return

    payload = {
        "smpl_pose": np.stack(smpl_pose_all),       # [N,21,3]
        "smpl_joints": np.stack(smpl_joints_all),   # [N,24,3]
        "body_quat": np.stack(body_quat_all),       # [N,4] (w,x,y,z)
        "frame_index": np.array(frame_idx_all, dtype=np.int64),
    }
    out_npz = os.path.join(args.out, "smpl_v2_payload.npz")
    np.savez(out_npz, **payload)

    # --- validacao do quaternion (norma ~1) ---
    quat_norms = np.linalg.norm(payload["body_quat"], axis=1)

    print("\n===== RESUMO STAGE 1 =====")
    print(f"frames processados : {frame_index}")
    print(f"smpl_pose shape    : {payload['smpl_pose'].shape}  (esperado [N,21,3])")
    print(f"smpl_joints shape  : {payload['smpl_joints'].shape}  (esperado [N,24,3])")
    print(f"body_quat shape    : {payload['body_quat'].shape}  (esperado [N,4], ordem w,x,y,z)")
    print(f"frame_index        : {payload['frame_index'].tolist()}")
    print(f"erro max roundtrip rotmat<->axis-angle : {max_roundtrip_err:.3e} (deve ser ~1e-6)")
    print(f"norma media dos quaternions            : {quat_norms.mean():.6f} (deve ser ~1.0)")
    print(f"payload salvo em   : {out_npz}")
    ok = (max_roundtrip_err < 1e-4) and (abs(quat_norms.mean() - 1.0) < 1e-3)
    print(f"GATE STAGE 1       : {'PASSOU' if ok else 'FALHOU'}")


if __name__ == "__main__":
    main()
