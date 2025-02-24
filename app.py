import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

def compute_flight_time(y0, v0, angle_deg, ay):
    """Compute flight time (when y returns to 0)."""
    theta = math.radians(angle_deg)
    if abs(ay) < 1e-6:
        return 10  # Arbitrary if no vertical acceleration.
    A = 0.5 * ay
    B = v0 * math.sin(theta)
    C = y0
    disc = B**2 - 4 * A * C
    if disc < 0:
        return 10
    t1 = (-B + math.sqrt(disc)) / (2 * A)
    t2 = (-B - math.sqrt(disc)) / (2 * A)
    t_final = max(t1, t2)
    return t_final if t_final > 0 else 10

def compute_trajectory(y0, v0, angle_deg, ax, ay, t_vals):
    """Compute the trajectory coordinates."""
    theta = math.radians(angle_deg)
    x = v0 * math.cos(theta) * t_vals + 0.5 * ax * t_vals**2
    y = y0 + v0 * math.sin(theta) * t_vals + 0.5 * ay * t_vals**2
    return x, y

def compute_velocity(v0, angle_deg, ax, ay, t):
    """Compute velocity components at time t."""
    theta = math.radians(angle_deg)
    vx = v0 * math.cos(theta) + ax * t
    vy = v0 * math.sin(theta) + ay * t
    return vx, vy

st.title("Projectile Motion with Velocity Vector Components")

# Sidebar for interactive parameter input
st.sidebar.header("Simulation Parameters")
y0    = st.sidebar.number_input("Initial Height (y0)", value=10.0, step=0.1)
ax_val = st.sidebar.number_input("Acceleration in x (ax)", value=0.0, step=0.1)
ay_val = st.sidebar.number_input("Acceleration in y (ay)", value=-9.8, step=0.1)
v0    = st.sidebar.number_input("Initial Speed (v0)", value=10.0, step=0.1)
angle = st.sidebar.number_input("Launch Angle (deg)", value=45.0, step=1.0)

# Compute the flight time
T_final = compute_flight_time(y0, v0, angle, ay_val)

# Time slider
t_val = st.sidebar.slider("Time", min_value=0.0, max_value=float(T_final), value=0.0, step=0.01)

# Generate the trajectory
t_vals = np.linspace(0, T_final, 300)
x_vals, y_vals = compute_trajectory(y0, v0, angle, ax_val, ay_val, t_vals)

# Compute current position and velocity
x_current, y_current = compute_trajectory(y0, v0, angle, ax_val, ay_val, np.array([t_val]))
x_current = x_current[0]
y_current = y_current[0]
vx, vy = compute_velocity(v0, angle, ax_val, ay_val, t_val)
scale = 0.5  # Adjust scaling for arrow display
arrow_dx = scale * vx
arrow_dy = scale * vy

# Create the Plotly figure.
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Trajectory"))
fig.add_trace(go.Scatter(x=[x_current], y=[y_current], mode="markers",
                         marker=dict(color="red", size=10), name="Body"))

# Full velocity vector arrow
fig.add_annotation(
    x=x_current + arrow_dx,
    y=y_current + arrow_dy,
    ax=x_current,
    ay=y_current,
    xref="x", yref="y",
    axref="x", ayref="y",
    showarrow=True,
    arrowhead=3,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="green",
    text=""
)
# Horizontal (x) component arrow.
fig.add_annotation(
    x=x_current + arrow_dx,
    y=y_current,
    ax=x_current,
    ay=y_current,
    xref="x", yref="y",
    axref="x", ayref="y",
    showarrow=True,
    arrowhead=3,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="magenta",
    text=""
)
# Vertical (y) component arrow.
fig.add_annotation(
    x=x_current,
    y=y_current + arrow_dy,
    ax=x_current,
    ay=y_current,
    xref="x", yref="y",
    axref="x", ayref="y",
    showarrow=True,
    arrowhead=3,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="cyan",
    text=""
)

fig.update_layout(
    title="Projectile Motion with Velocity Vector Components",
    xaxis_title="Distance (m)",
    yaxis_title="Height (m)",
    xaxis=dict(range=[min(x_vals)*1.1, max(x_vals)*1.1]),
    yaxis=dict(range=[min(y_vals)*1.1, max(y_vals)*1.1]),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
