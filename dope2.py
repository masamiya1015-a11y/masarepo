#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Mon Oct 26 09:54:11 2026

#@author: /home/taizo/my_apps/dope2.py

import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import streamlit.components.v1 as components
from numba import jit



# 定数
G = 9.81

st.title("Double Pendulum Simulator")

# サイドバー：パラメータ入力
st.sidebar.header("Parameters")
L1 = st.sidebar.number_input("L1 (m)", value=1.0, format="%.2f")
L2 = st.sidebar.number_input("L2 (m)", value=0.5, format="%.2f")
M1 = st.sidebar.number_input("M1 (kg)", value=1.0, format="%.2f")
M2 = st.sidebar.number_input("M2 (kg)", value=0.5, format="%.2f")

yz1 = st.sidebar.number_input(" θ1 (π)", value=1.0, format="%.2f")
dyz1 = st.sidebar.number_input(" θ1 (π/s)", value=0.0, format="%.2f") 
yz2 = st.sidebar.number_input(" θ2 (π)", value=0.5, format="%.2f") 
dyz2 = st.sidebar.number_input(" θ2 (π/s)", value=0.0, format="%.2f") 
# 運動方程式の定義
@jit(nopython=True ,cache=True)
def equations(t, yt_set):
    yt1, dyt1, yt2, dyt2 = yt_set
    delta = yt1 - yt2

    num1 = -G * (2 * M1 + M2) * np.sin(yt1) - G * M2 * np.sin(yt1 - 2 * yt2) \
           - 2 * np.sin(delta) * M2 * (dyt2**2 * L2 +
                                       dyt1**2 * L1 * np.cos(delta))
    den1 = L1 * (2 * M1 + M2 - M2 * np.cos(2 * delta))
    ddyt1 = num1 / den1

    num2 = 2 * np.sin(delta) * (dyt1**2 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(yt1)
                                + dyt2**2 * L2 * M2 * np.cos(delta))
    den2 = L2 * (2 * M1 + M2 - M2 * np.cos(2 * delta))
    ddyt2 = num2 / den2

    return [dyt1, ddyt1, dyt2, ddyt2]


# 初期状態と時間設定
#initial_state = [np.pi / 2, 0, np.pi, 0]
initial_state = [np.pi*yz1, np.pi*dyz1, np.pi*yz2, np.pi*dyz2]
t_max = 20.0
t_span = [0, t_max]
t_eval = np.linspace(0, t_max, 300)

# 微分方程式の数値計算
solution = solve_ivp(equations, t_span, initial_state,
                     t_eval=t_eval, method="RK45")

# 角度データから直交座標(x, y)への変換
yt1, yt2 = solution.y[0], solution.y[2]
x1 = L1 * np.sin(yt1)
y1 = -L1 * np.cos(yt1)
x2 = x1 + L2 * np.sin(yt2)
y2 = y1 - L2 * np.cos(yt2)

# アニメーション用のフィギュア設定
fig, ax = plt.subplots(figsize=(6, 6))
max_len = L1 + L2 + 0.5
ax.set_xlim(-max_len, max_len)
ax.set_ylim(-max_len, max_len)
ax.set_aspect('equal')
ax.grid(True)

line, = ax.plot([], [], "o-", lw=2, color="#1f77b4")
trace, = ax.plot([], [], '-', lw=1, color="#ff7f0e", alpha=0.6)


def update(i):
    this_x = [0, x1[i], x2[i]]
    this_y = [0, y1[i], y2[i]]
    line.set_data(this_x, this_y)
    trace.set_data(x2[:i], y2[:i])
    return line, trace


# アニメーションの作成
ani = animation.FuncAnimation(
    fig, update, frames=len(solution.t), interval=40, blit=True)
plt.close(fig)  # Streamlit上での静止画の重複描画を防ぐ

# アニメーションをHTMLに変換してStreamlitに表示
st.subheader("Simulation Result")
html_to_display = ani.to_jshtml()
components.html(html_to_display, height=700)
