import streamlit as st
import numpy as np
from PIL import Image
import math
import io

st.set_page_config(page_title="シャインエフェクトGIFメーカー", layout="centered")

st.title("✨ シャインエフェクトGIFメーカー")
st.write("画像をアップロードして、高級感のあるシャインエフェクトGIFを作成します。")

# 1. 画像アップロード
uploaded_file = st.file_uploader("画像をアップロードしてください (JPG/PNG)", type=['png', 'jpg', 'jpeg'])

# 2. パラメータ調整スライダー
st.subheader("⚙️ 光の調整")
col1, col2 = st.columns(2)
with col1:
    # 帯の広さ（ふんわり感）
    band_width_ratio = st.slider("光の幅（ふんわり感）", min_value=0.1, max_value=0.8, value=0.45, step=0.05)
with col2:
    # 光の強さ（白飛び防止）
    intensity_ratio = st.slider("光の強さ（明るさ）", min_value=0.01, max_value=0.30, value=0.08, step=0.01)

# 3. GIF生成処理
if uploaded_file is not None:
    # 元画像のプレビュー
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="元画像", use_container_width=True)
    
    if st.button("GIFを生成する", type="primary"):
        with st.spinner("生成中...（数秒かかります）"):
            # アニメーション設定
            target_width = 300
            orig_w, orig_h = image.size
            target_height = int(orig_h * (target_width / orig_w))
            img_resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            h, w, _ = img_array.shape
            frames = []
            
            angle_deg = 35 
            theta = math.radians(angle_deg)
            u_min = -w * 0.6
            u_max = w * math.cos(theta) + h * math.sin(theta) + w * 0.6
            num_frames = 60
            
            band_width = w * band_width_ratio
            highlight_width = w * (band_width_ratio * 0.18) 
            
            for i in range(num_frames):
                t = i / (num_frames - 1)
                eased_t = t ** 1.5 
                current_u = u_min + (u_max - u_min) * eased_t
                
                fade = 1.0
                if t < 0.15: fade = t / 0.15
                elif t > 0.85: fade = (1.0 - t) / 0.15
                
                y_indices, x_indices = np.indices((h, w))
                u_grid = x_indices * math.cos(theta) + y_indices * math.sin(theta)
                dist = np.abs(u_grid - current_u)
                
                band_intensity = np.exp(- (dist / band_width)**2) * intensity_ratio
                highlight_intensity = np.exp(- (dist / highlight_width)**2) * (intensity_ratio * 1.8)
                
                total_intensity = (band_intensity + highlight_intensity) * fade
                intensity_3d = np.repeat(total_intensity[:, :, np.newaxis], 3, axis=2)
                
                result_array = 1.0 - (1.0 - img_array) * (1.0 - intensity_3d)
                result_img = np.clip(result_array * 255.0, 0, 255).astype(np.uint8)
                frames.append(Image.fromarray(result_img))
            
            # メモリ上でGIFを保存（直接ダウンロードさせるため）
            gif_io = io.BytesIO()
            frames[0].save(gif_io, format='GIF', save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=False)
            gif_bytes = gif_io.getvalue()
            
            st.success("生成完了！")
            
            # GIFの表示
            st.image(gif_bytes, caption="完成したGIFアニメーション")
            
            # ダウンロードボタン
            st.download_button(
                label="📥 GIFをダウンロード",
                data=gif_bytes,
                file_name="shine_effect.gif",
                mime="image/gif"
            )