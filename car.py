from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import List
import cv2
import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 页面与样式 -------------------------------------------------------------------
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="行车记录仪记录仪视觉增强系统",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# 行车记录仪视觉增强系统 v1.0\n基于YOLOv8的交通目标识别解决方案",
    },
)

# -------------------- 全局 SessionState 初始化 -------------------------------
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "detection_frames" not in st.session_state:
    st.session_state.detection_frames: List[np.ndarray] = []
if "output_video_path" not in st.session_state:
    st.session_state.output_video_path: str | None = None
if "video_info" not in st.session_state:
    st.session_state.video_info = {
        "width": 0,
        "height": 0,
        "fps": 0,
        "duration": 0.0,
        "frame_count": 0,
    }
if "class_ids" not in st.session_state:
    st.session_state.class_ids = [2, 3, 5, 7]

# -----------------------------------------------------------------------------
# 样式（保持不变） -------------------------------------------------------------
# -----------------------------------------------------------------------------
CUSTOM_STYLE = """
<style>
    .stProgress > div > div > div > div {background-color: #1f77b4;}
    .stButton>button {background-color: #4CAF50; color:#fff; font-weight:bold; border-radius:5px; padding:0.5rem 1rem; border:none;}
    .stButton>button:hover {background:#45a049;}
    .stDownloadButton>button {background:#2196F3; color:#fff; font-weight:bold; border-radius:5px; padding:0.5rem 1rem; border:none;}
    .stDownloadButton>button:hover {background:#0b7dda;}
    .header-style {color:#2c3e50; border-bottom:2px solid #3498db; padding-bottom:0.2rem;}
    .metric-card {background:#f8f9fa; border-radius:10px; padding:15px; box-shadow:0 4px 6px rgba(0,0,0,0.1);}
</style>
"""
st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 侧边栏：模型与视频上传 -------------------------------------------------------
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("行车记录仪视觉增强系统")
    st.divider()

    st.subheader("模型设置")

    # 模型缓存
    @st.cache_resource(show_spinner="正在加载模型…")
    def load_model(weights_path: str) -> YOLO:  # noqa: D401
        """加载并缓存 YOLO 模型 (会话级)."""
        return YOLO(weights_path)

    # 模型选择
    model_file = st.file_uploader("上传 YOLOv8 模型文件 (.pt)", type=["pt"], key="model_uploader")
    if model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            tmp_file.write(model_file.read())
            model_path = tmp_file.name
        try:
            st.session_state.model = load_model(model_path)
            st.session_state.model_loaded = True
            st.success("模型加载成功")
        except Exception as exc:  # noqa: BLE001
            st.error(f"模型加载失败: {exc}")
            st.session_state.model_loaded = False
    else:
        st.info("请上传 YOLOv8 模型文件")

    # -------------------- 视频上传 -------------------------------------------
    st.divider()
    st.subheader("视频上传")
    video_file = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov"], key="video_uploader")
    if video_file is not None:
        st.session_state.video_file = video_file
        st.session_state.video_name = video_file.name

        # 读取视频基本信息
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_cnt / fps if fps else 0.0
            st.session_state.video_info.update(
                {
                    "width": w,
                    "height": h,
                    "fps": fps,
                    "duration": duration,
                    "frame_count": frame_cnt,
                }
            )
            st.success(f"视频已上传: {video_file.name}")
            st.info(f"分辨率 {w}×{h} | {fps:.1f} FPS | 时长 {duration:.1f} 秒")
        else:
            st.error("无法读取视频信息")
        cap.release()
        os.unlink(video_path)
    else:
        st.info("请上传视频文件")

    # -------------------- 检测参数 -------------------------------------------
    st.divider()
    st.subheader("检测参数")
    st.slider("置信度阈值", 0.0, 1.0, 0.5, 0.05, key="confidence_threshold")
    st.slider("IoU 阈值", 0.0, 1.0, 0.7, 0.05, key="iou_threshold")
    st.slider("帧跳过 (加速处理)", 1, 10, 3, 1, key="frame_skip")

    # -------------------- 检测类别 -------------------------------------------
    st.subheader("检测类别")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        chk_car = st.checkbox("汽车", True)
        chk_truck = st.checkbox("卡车", True)
        chk_bus = st.checkbox("公交车", True)
    with col_c2:
        chk_motor = st.checkbox("摩托车", True)
        chk_person = st.checkbox("行人", False)
        chk_light = st.checkbox("交通灯", False)

    class_ids: list[int] = []
    if chk_car:
        class_ids.extend([2, 7])  # car & truck
    if chk_truck and 7 not in class_ids:
        class_ids.append(7)
    if chk_bus:
        class_ids.append(5)
    if chk_motor:
        class_ids.append(3)
    if chk_person:
        class_ids.append(0)
    if chk_light:
        class_ids.append(9)

    # 开始检测按钮 ------------------------------------------------------------
    if st.button(
        "开始检测",
        disabled=not (st.session_state.model_loaded and video_file is not None),
    ):
        st.session_state.processing = True
        st.session_state.video_processed = False
        st.session_state.detection_frames.clear()
        st.session_state.output_video_path = None
        st.session_state.class_ids = class_ids

# -----------------------------------------------------------------------------
# 主体布局：原始视频 / 检测结果 ------------------------------------------------
# -----------------------------------------------------------------------------
col_raw, col_result = st.columns(2)

with col_raw:
    st.markdown("<h3 class='header-style'>原始视频</h3>", unsafe_allow_html=True)
    if video_file is not None:
        st.video(video_file)
    else:
        st.info("上传视频后将在此处显示")

# -----------------------------------------------------------------------------
# 视频处理函数 (核心优化) -----------------------------------------------------
# -----------------------------------------------------------------------------

def process_video():  # noqa: C901
    """执行目标检测并写出新视频，同时抽样 4 帧做缩略图."""
    if not (st.session_state.model_loaded and "video_file" in st.session_state):
        return

    # --- 保存上传视频到临时文件 ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(st.session_state.video_file.getvalue())
        src_path = tmp.name

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        st.error("无法打开视频文件")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- 输出文件 ---
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    dst_path = out_file.name
    out_file.close()
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))

    # 检测辅助工具
    tracker = sv.ByteTrack()
    box_anno = sv.BoundingBoxAnnotator()
    label_anno = sv.LabelAnnotator()

    # 抽样帧索引 (10% 40% 70% 90%)
    sample_idx_set = {int(total * r) for r in [0.1, 0.4, 0.7, 0.9]}

    progress_bar = st.progress(0.0)
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 帧跳过
        if frame_no % st.session_state.get("frame_skip", 3) != 0:
            frame_no += 1
            continue

        # 模型推理
        results = st.session_state.model.predict(
            frame,
            conf=st.session_state.get("confidence_threshold", 0.5),
            iou=st.session_state.get("iou_threshold", 0.7),
            classes=st.session_state.class_ids,
            verbose=False,
        )[0]

        dets = sv.Detections.from_ultralytics(results)
        dets = tracker.update_with_detections(dets)

        labels = [
            f"{st.session_state.model.names[c]} {conf:.2f}"
            for c, conf in zip(dets.class_id, dets.confidence)
        ]

        annotated = frame.copy()
        annotated = box_anno.annotate(annotated, dets)
        annotated = label_anno.annotate(annotated, dets, labels)

        writer.write(annotated)

        # 仅在 sample_idx_set 中保留缩略图
        if frame_no in sample_idx_set:
            st.session_state.detection_frames.append(annotated.copy())

        frame_no += 1
        progress_bar.progress(min(frame_no / total, 1.0))

    # 结束
    cap.release()
    writer.release()
    os.unlink(src_path)

    st.session_state.output_video_path = dst_path
    st.session_state.video_processed = True
    st.session_state.processing = False
    progress_bar.empty()

# ------------------ 如果正在处理，后台线程执行 -------------------------------
if st.session_state.processing:
    with st.spinner("视频处理中，请稍候…"):
        process_video()

# -----------------------------------------------------------------------------
# 结果展示 ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
with col_result:
    st.markdown("<h3 class='header-style'>增强结果</h3>", unsafe_allow_html=True)
    if st.session_state.video_processed and st.session_state.output_video_path:
        with open(st.session_state.output_video_path, "rb") as fvid:
            st.video(fvid.read(), format="video/mp4")
        st.success("视频处理完成")

        # 下载按钮
        with open(st.session_state.output_video_path, "rb") as fvid:
            st.download_button(
                "下载增强视频",
                data=fvid.read(),
                file_name=f"enhanced_{st.session_state.video_name}",
                mime="video/mp4",
            )
    elif st.session_state.processing:
        st.info("正在处理视频，请稍候…")
    else:
        st.info("处理结果将在此处显示")

# 缩略图 & 统计信息 -----------------------------------------------------------
if st.session_state.video_processed:
    st.divider()
    st.markdown("<h3 class='header-style'>处理样本</h3>", unsafe_allow_html=True)
    cols_thumb = st.columns(4)
    for i, img in enumerate(st.session_state.detection_frames):
        cols_thumb[i].image(img, channels="BGR", use_column_width=True)

    # ----- 统计卡片 ----------------------------------------------
    st.divider()
    st.markdown("<h3 class='header-style'>处理统计</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='metric-card'><h4>视频信息</h4><p>分辨率: {st.session_state.video_info['width']}×{st.session_state.video_info['height']}</p><p>帧率: {st.session_state.video_info['fps']:.1f} FPS</p><p>时长: {st.session_state.video_info['duration']:.1f} 秒</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        speed = len(st.session_state.detection_frames) / max(st.session_state.video_info["duration"], 1)
        st.markdown(
            f"<div class='metric-card'><h4>处理信息</h4><p>检测帧数: {len(st.session_state.detection_frames)}</p><p>帧跳过: {st.session_state.get('frame_skip', 3)}</p><p>处理速度: {speed:.1f} FPS</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'><h4>检测参数</h4><p>置信度: {st.session_state.get('confidence_threshold', 0.5)}</p><p>IoU 阈值: {st.session_state.get('iou_threshold', 0.7)}</p><p>类别数: {len(st.session_state.class_ids)}</p></div>",
            unsafe_allow_html=True,
        )
    with c4:
        out_name = Path(st.session_state.output_video_path).name if st.session_state.output_video_path else "-"
        st.markdown(
            f"<div class='metric-card'><h4>系统信息</h4><p>模型状态: 已加载</p><p>处理状态: 完成</p><p>输出文件: {out_name}</p></div>",
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# 使用说明 & 页脚 -----------------------------------------------------------
# -----------------------------------------------------------------------------

st.divider()
st.markdown("<h3 class='header-style'>使用说明</h3>", unsafe_allow_html=True)

st.markdown(
    """
<div style="background-color:#f0f7ff;padding:20px;border-radius:10px;border-left:4px solid #3498db;">
<ol>
<li><strong>上传模型</strong>：在左侧边栏上传您的 YOLOv8 模型文件（.pt 格式）</li>
<li><strong>上传视频</strong>：上传要处理的行车记录仪视频文件（MP4/AVI/MOV）</li>
<li><strong>调整参数</strong>：设置置信度、IoU 和帧跳过等参数</li>
<li><strong>开始处理</strong>：点击“开始检测”按钮进行处理</li>
<li><strong>查看结果</strong>：处理完成后，右侧将播放增强视频；下方展示处理样本</li>
<li><strong>下载结果</strong>：点击下载按钮将增强后的视频保存到本地</li>
</ol>
<p><strong>提示</strong>：首次运行时模型加载可能较慢；处理高清视频需较高算力。</p>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()
st.markdown(
    """
<div style="text-align:center;padding:20px;color:#7f8c8d;">
    行车记录仪视觉增强系统 v1.1 © 2025
</div>
""",
    unsafe_allow_html=True,
)