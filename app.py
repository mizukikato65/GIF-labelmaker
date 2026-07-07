import streamlit as st
import numpy as np
from PIL import Image
import math
import io

# ページ設定 (絵文字なし)
st.set_page_config(page_title="商品ラベルGIFジェネレーター", layout="centered")

# タイトルとリード文 (絵文字なし)
st.title("商品ラベルGIFジェネレーター")
st.markdown("""
商品ラベルに装飾エフェクトを追加できます。  
※現在追加できるのは一種類のみです。
""")

# --- 設定エリア ---
st.sidebar.header("エフェクト設定")

# 1. 光の質感設定
band_width_ratio = st.sidebar.slider("光の幅（ふんわり感）", min_value=0.1, max_value=0.8, value=0.45, step=0.05)
intensity_ratio = st.sidebar.slider("光の強さ（明るさ）", min_value=0.01, max_value=0.30, value=0.08, step=0.01)

# 2. 速度設定
speed_val = st.sidebar.select_slider(
    "光の流れる速度",
    options=[100, 80, 60, 40, 20],
    value=40,
    format_func=lambda x: "遅い" if x == 100 else ("速い" if x == 20 else "標準")
)

# 3. 画像アップロード
uploaded_file = st.file_uploader("ラベル画像をアップロード (JPG/PNG)", type=['png', 'jpg', 'jpeg'])

def process_frame(base_img_array, t, band_w, high_w, intensity, theta, u_min, u_max):
    """特定の進行度(t)のフレームを計算する関数"""
    h, w, _ = base_img_array.shape
    eased_t = t ** 1.5 
    current_u = u_min + (u_max - u_min) * eased_t
    
    # ループの端でのフェード
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
    
    # スクリーン合成
    result_array = 1.0 - (1.0 - base_img_array) * (1.0 - intensity_3d)
    return np.clip(result_array * 255.0, 0, 255).astype(np.uint8)

def generate_gif(img_array, num_frames, band_w, high_w, intensity, theta, u_min, u_max, duration):
    """GIFを生成してバイトデータを返す関数"""
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        frame_data = process_frame(img_array, t, band_w, high_w, intensity, theta, u_min, u_max)
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

if uploaded_file is not None:
    # 画像の読み込み
    image = Image.open(uploaded_file).convert('RGB')
    
    # 計算用リサイズ
    target_width = 300
    orig_w, orig_h = image.size
    target_height = int(orig_h * (target_width / orig_w))
    img_resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    
    # アニメーション基本パラメータ
    h, w, _ = img_array.shape
    angle_deg = 35 
    theta = math.radians(angle_deg)
    u_min = -w * 0.6
    u_max = w * math.cos(theta) + h * math.sin(theta) + w * 0.6
    
    band_w = w * band_width_ratio
    high_w = w * (band_width_ratio * 0.18)

    # --- 仕上がりプレビュー (動的GIFプレビュー) ---
    st.subheader("仕上がりプレビュー")
    
    # プレビュー用GIF生成 (スライダー変更で再実行される)
    # パフォーマンスを考慮し、プレビューのフレーム数を少し減らす（例: 30フレーム）ことも検討可能
    # ここでは60フレームのまま実装
    preview_gif_bytes = generate_gif(
        img_array, num_frames=60, band_w=band_w, high_w=high_w, 
        intensity=intensity_ratio, theta=theta, u_min=u_min, u_max=u_max, 
        duration=speed_val
    )
    
    st.image(preview_gif_bytes, caption="仕上がりプレビュー（GIF）", width=300)
    st.info("左側のスライダーを動かすとプレビューが自動更新されます。")

    # --- 本番生成 ---
    if st.button("GIFを最終生成する", type="primary"):
        # プレビューで生成されたGIFデータを再利用する
        gif_bytes = preview_gif_bytes
        
        st.download_button(
            label="GIFをダウンロード",
            data=gif_bytes,
            file_name="product_label_shine.gif",
            mime="image/gif"
        )
