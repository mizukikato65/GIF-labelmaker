import streamlit as st
import numpy as np
from PIL import Image
import math
import io

# ページ設定 (絵文字なし)
st.set_page_config(page_title="商品ラベルGIFジェネレーター", layout="centered")

# タイトルとリード文
st.title("商品ラベルGIFジェネレーター")
st.markdown("""
商品ラベルに装飾エフェクトを追加できます。  
左のメニューからエフェクトの種類を選択してください。
""")

# --- 設定エリア ---
st.sidebar.header("エフェクト設定")

# エフェクトの種類の選択
effect_type = st.sidebar.selectbox(
    "エフェクトの種類",
    ["シャイン（斜めの光）", "スポットライト（円形の光）", "パルス（全体の発光）"]
)

st.sidebar.markdown("---")

# 光の質感設定
band_width_ratio = st.sidebar.slider("光の幅（ふんわり感）", min_value=0.1, max_value=0.8, value=0.45, step=0.05)
intensity_ratio = st.sidebar.slider("光の強さ（明るさ）", min_value=0.01, max_value=0.30, value=0.08, step=0.01)

# 速度設定
speed_val = st.sidebar.select_slider(
    "アニメーションの速度",
    options=[100, 80, 60, 40, 20],
    value=40,
    format_func=lambda x: "遅い" if x == 100 else ("速い" if x == 20 else "標準")
)

# 画像アップロード
uploaded_file = st.file_uploader("ラベル画像をアップロード (JPG/PNG)", type=['png', 'jpg', 'jpeg'])

# --- 各エフェクトのフレーム計算関数 ---

def process_frame_shine(base_img_array, t, band_w, high_w, intensity, theta, u_min, u_max):
    """シャイン（斜めの光）の計算"""
    h, w, _ = base_img_array.shape
    eased_t = t ** 1.5 
    current_u = u_min + (u_max - u_min) * eased_t
    
    fade = 1.0
    if t < 0.15: fade = t / 0.15
    elif t > 0.85: fade = (1.0 - t) / 0.15
    
    y_indices, x_indices = np.indices((h, w))
    u_grid = x_indices * math.cos(theta) + y_indices * math.sin(theta)
    dist = np.abs(u_grid - current_u)
    
    band_intensity = np.exp(- (dist / band_w)**2) * intensity
    highlight_intensity = np.exp(- (dist / high_w)**2) * (intensity * 1.8)
    
    total_intensity = (band_intensity + highlight_intensity) * fade
    intensity_3d = np.repeat(total_intensity[:, :, np.newaxis], 3, axis=2)
    
    result_array = 1.0 - (1.0 - base_img_array) * (1.0 - intensity_3d)
    return np.clip(result_array * 255.0, 0, 255).astype(np.uint8)

def process_frame_spotlight(base_img_array, t, band_w, intensity):
    """スポットライト（円形の光）の計算"""
    h, w, _ = base_img_array.shape
    
    # イージングと移動範囲 (左から右へ水平移動)
    eased_t = t ** 1.5
    current_x = -w * 0.4 + (w * 1.8) * eased_t
    current_y = h / 2.0
    
    fade = 1.0
    if t < 0.15: fade = t / 0.15
    elif t > 0.85: fade = (1.0 - t) / 0.15
    
    y_indices, x_indices = np.indices((h, w))
    # 中心からの距離を計算
    dist = np.sqrt((x_indices - current_x)**2 + (y_indices - current_y)**2)
    
    spot_intensity = np.exp(- (dist / band_w)**2) * (intensity * 1.5) * fade
    intensity_3d = np.repeat(spot_intensity[:, :, np.newaxis], 3, axis=2)
    
    result_array = 1.0 - (1.0 - base_img_array) * (1.0 - intensity_3d)
    return np.clip(result_array * 255.0, 0, 255).astype(np.uint8)

def process_frame_pulse(base_img_array, t, intensity):
    """パルス（全体の発光）の計算"""
    # サイン波を使って、0 -> 1 -> 0 へと滑らかに変化させる
    cycle = math.sin(t * math.pi)
    
    # パルスは全体が明るくなるため、強さを少し抑える
    pulse_intensity = (intensity * 0.8) * cycle
    intensity_3d = np.full_like(base_img_array, pulse_intensity)
    
    result_array = 1.0 - (1.0 - base_img_array) * (1.0 - intensity_3d)
    return np.clip(result_array * 255.0, 0, 255).astype(np.uint8)

def generate_gif(img_array, num_frames, effect_name, band_w, high_w, intensity, theta, u_min, u_max, duration):
    """選択されたエフェクトに応じてGIFを生成する関数"""
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        if effect_name == "シャイン（斜めの光）":
            frame_data = process_frame_shine(img_array, t, band_w, high_w, intensity, theta, u_min, u_max)
        elif effect_name == "スポットライト（円形の光）":
            frame_data = process_frame_spotlight(img_array, t, band_w, intensity)
        elif effect_name == "パルス（全体の発光）":
            frame_data = process_frame_pulse(img_array, t, intensity)
            
        frames.append(Image.fromarray(frame_data))
    
    gif_io = io.BytesIO()
    frames[0].save(
        gif_io, 
        format='GIF', 
        save_all=True, 
        append_images=frames[1:], 
        duration=duration,
        loop=0, 
        optimize=False
    )
    return gif_io.getvalue()

# --- メイン処理 ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    target_width = 300
    orig_w, orig_h = image.size
    target_height = int(orig_h * (target_width / orig_w))
    img_resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    
    h, w, _ = img_array.shape
    angle_deg = 35 
    theta = math.radians(angle_deg)
    u_min = -w * 0.6
    u_max = w * math.cos(theta) + h * math.sin(theta) + w * 0.6
    
    band_w = w * band_width_ratio
    high_w = w * (band_width_ratio * 0.18)

    st.subheader("仕上がりプレビュー")
    
    # 選択されたエフェクトタイプを渡してプレビュー生成
    preview_gif_bytes = generate_gif(
        img_array, num_frames=60, effect_name=effect_type, 
        band_w=band_w, high_w=high_w, intensity=intensity_ratio, 
        theta=theta, u_min=u_min, u_max=u_max, duration=speed_val
    )
    
    st.image(preview_gif_bytes, caption="仕上がりプレビュー（GIF）", width=300)
    st.info("左側のスライダーやエフェクト種類を変更するとプレビューが自動更新されます。")

    if st.button("GIFをダウンロード", type="primary"):
        st.download_button(
            label="ダウンロードを実行",
            data=preview_gif_bytes,
            file_name=f"label_effect_{effect_type[:4]}.gif",
            mime="image/gif"
        )
