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
    ["シャイン（斜めの光）", "スポットライト（円形の光）", "パルス（全体の発光）", "キラキラ光る（不規則な点）"]
)

st.sidebar.markdown("---")

# 光の質感設定
band_width_ratio = st.sidebar.slider("光の幅（ふんわり感/点サイズ）", min_value=0.1, max_value=0.8, value=0.45, step=0.05)
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
    eased_t = t ** 1.5
    current_x = -w * 0.4 + (w * 1.8) * eased_t
    current_y = h / 2.0
    
    fade = 1.0
    if t < 0.15: fade = t / 0.15
    elif t > 0.85: fade = (1.0 - t) / 0.15
    
    y_indices, x_indices = np.indices((h, w))
    dist = np.sqrt((x_indices - current_x)**2 + (y_indices - current_y)**2)
    
    spot_intensity = np.exp(- (dist / band_w)**2) * (intensity * 1.5) * fade
    intensity_3d = np.repeat(spot_intensity[:, :, np.newaxis], 3, axis=2)
    
    result_array = 1.0 - (1.0 - base_img_array) * (1.0 - intensity_3d)
    return np.clip(result_array * 255.0, 0, 255).astype(np.uint8)

def process_frame_pulse(base_img_array, t, intensity):
    """パルス（全体の発光）の計算"""
    cycle = math.sin(t * math.pi)
    pulse_intensity = (intensity * 0.8) * cycle
    intensity_3d = np.full_like(base_img_array, pulse_intensity)
    
    result_array = 1.0 - (1.0 - base_img_array) * (1.0 - intensity_3d)
    return np.clip(result_array * 255.0, 0, 255).astype(np.uint8)

def process_frame_kirakira(base_img_array, t, band_w, intensity):
    """キラキラ（滑らかな明滅）の計算"""
    h, w, _ = base_img_array.shape
    
    # 乱数シードを固定し、全フレームでキラキラの位置と明滅のタイミングを統一する
    np.random.seed(42)
    
    num_sparkles = 40
    # キラキラの固定位置
    positions = np.random.rand(num_sparkles, 2)
    positions[:, 0] *= w
    positions[:, 1] *= h
    
    # 位相（明滅のスタート地点）をランダムにずらす
    phases = np.random.rand(num_sparkles) * 2 * math.pi
    # アニメーションループが完璧に繋がるよう、明滅の速さは整数倍（1〜3回）に設定
    speeds = np.random.randint(1, 4, size=num_sparkles)
    
    y_indices, x_indices = np.indices((h, w))
    sparkle_intensity = np.zeros_like(x_indices, dtype=np.float32)
    
    for i in range(num_sparkles):
        dist = np.sqrt((x_indices - positions[i, 0])**2 + (y_indices - positions[i, 1])**2)
        
        # サイン波を利用して 0.0 〜 1.0 を滑らかに行き来させる
        twinkle = (math.sin(t * 2 * math.pi * speeds[i] + phases[i]) + 1.0) / 2.0
        
        # キラキラのサイズと明るさ
        sparkle_spot_intensity = np.exp(- (dist / (band_w * 0.5))**2) * (intensity * 2.5) * twinkle
        sparkle_intensity += sparkle_spot_intensity
        
    sparkle_intensity = np.clip(sparkle_intensity, 0, 1.0)
    intensity_3d = np.repeat(sparkle_intensity[:, :, np.newaxis], 3, axis=2)
    
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
        elif effect_name == "キラキラ光る（不規則な点）":
            frame_data = process_frame_kirakira(img_array, t, band_w * 0.4, intensity * 2.0)
            
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
    
    # プレビュー生成（キラキラの場合も滑らかさを確認できるよう60フレームに戻しています）
    preview_gif_bytes = generate_gif(
        img_array, num_frames=60, effect_name=effect_type, 
        band_w=band_w, high_w=high_w, intensity=intensity_ratio, 
        theta=theta, u_min=u_min, u_max=u_max, duration=speed_val
    )
    
    st.image(preview_gif_bytes, caption="仕上がりプレビュー（GIF）", width=300)
    st.info("左側のスライダーやエフェクト種類を変更するとプレビューが自動更新されます。")

    # そのまま即座にダウンロードできるようにUIを調整
    st.download_button(
        label="完成したGIFをダウンロード",
        data=preview_gif_bytes,
        file_name=f"label_effect_{effect_type[:4]}.gif",
        mime="image/gif",
        type="primary"
    )
