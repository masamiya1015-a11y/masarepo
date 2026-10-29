#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:21:02 2026

@author: Julia streamlit num & char input type app3.py
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
@jit(nopython=True)
def julia_set(n, z, c):

    for ii in range(n):
        z = z*z + c  
        if abs(z) >= 2.0 :
            return ii
    return n   

# --- Streamlitの画面設定 ---
st.title("ジュリア集合 シミュレーター")
st.write("左側のサイドバーでパラメータを変更すると、自動的に再計算されて描画されます。")

# --- サイドバーに入力項目（スライダーや数値入力）を配置 ---
st.sidebar.header("パラメータ設定")

# 1. 複素数 c の設定（実部と虚部を個別にスライダーで調整できるようにしました）
c_real = st.sidebar.number_input(
    "複素数c(Real)", value=-0.4062, format="%.4f")
c_imag = st.sidebar.number_input("複素数c(Imag)" , value=-0.9375, format = "%.4f")

cc = complex(c_real, c_imag)

# 2. 表示中心座標 (x_c, y_c)
x_c = st.sidebar.number_input("中心 X  (x_c)" , value=0.0,format= "%.2f") 
y_c = st.sidebar.number_input("中心 Y  (y_c)" , value=0.0, format="%.2f")

# 3. 画面の幅と解像度
width = st.sidebar.number_input("表示幅 (width)", value=4.0,format="%.2f")
resolution = st.sidebar.number_input("解像度 (値が大　高精細)" , value=500) 

nn = st.sidebar.number_input("回数 (nn)" , value=100)  


name_set = ["kana","dai","syuu","kou","yumi","masa" ]
select_name = st.sidebar.selectbox("カラーマップ選択", name_set)

ctable = {"kana":"viridis","dai":"plasma","syuu":"inferno","kou":"magma","yumi":"hot","masa":"copper" }

scmap = ctable.get(select_name)


frm_size = 25
height = width * 9 / 16
xdevn = resolution * 16 // frm_size
ydevn = resolution *  9 // frm_size
x_start = x_c - width / 2

x_end   = x_c + width / 2
y_start = y_c - height/ 2
y_end   = y_c + height/ 2
xdevn = int(xdevn)
ydevn = int(ydevn)

x = np.linspace(x_start, x_end, xdevn)
y = np.linspace(y_start, y_end, ydevn)

# 計算中であることを画面に示す
with st.spinner("計算中... しばらくお待ちください"):
    mesh = [[0 for j in range(xdevn)] for i in range(ydevn)]

    for i in range(ydevn): 
        for j in range(xdevn):
            zz = complex(x[j], y[i])
            # 元の関数を呼び出し
            inn = julia_set(nn, zz, cc)
            mesh[i][j] = inn


fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot([-2, 2], [0, 0], color="blue", alpha=0.5)
ax.plot([0, 0], [-1, 1], color="blue", alpha=0.5)

ax.plot(cc.real, cc.imag, "r*", markersize=8)
ax.set_xlabel("Real")
ax.set_ylabel("Image")
extent = (x_start, x_end, y_start, y_end)
#ax.imshow(mesh, extent=extent, cmap=colormap)
ax.imshow(mesh, extent=extent, cmap= scmap)
# 元のコードにあったテキスト表示をStreamlit風に綺麗に表示
st.write(f"**現在のパラメータ:** c = {cc} | 中心 = ({x_c}, {y_c}) | 幅 = {width}")

# グラフをStreamlitの画面に出力
st.pyplot(fig)