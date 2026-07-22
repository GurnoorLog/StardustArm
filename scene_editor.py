import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gradio as gr

SCENE_PATH = os.path.join(
    os.path.dirname(__file__),
    "assets", "robots", "mephi_arm", "scene.xml"
)
PRESETS_DIR = os.path.join(os.path.dirname(__file__), "scene_presets")

BALL_NAMES = ["ball_primary", "ball_secondary"]
OBSTACLE_NAMES = ["obstacle1", "obstacle2", "obstacle3"]


def parse_positions(xml_path=None):
    if xml_path is None:
        xml_path = SCENE_PATH
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    def get_position(element, name):
        if element is not None and element.get("name") == name:
            pos_str = element.get("pos", "0 0 0")
            return [float(v) for v in pos_str.split()]
        return None

    positions = {}
    for child in worldbody.iter("body"):
        name = child.get("name", "")
        pos_str = child.get("pos", "0 0 0")
        positions[name] = [float(v) for v in pos_str.split()]

    return positions


def set_position(xml_path, body_name, pos):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    for child in worldbody.iter("body"):
        if child.get("name") == body_name:
            child.set("pos", f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
            break
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def render_preview(xml_path=None):
    if xml_path is None:
        xml_path = SCENE_PATH
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.azimuth = 120
    cam.elevation = -20
    cam.distance = 1.5
    cam.lookat[:] = [0.0, 0.0, 0.1]

    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, 640, 480), scene, ctx)

    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, 640, 480), ctx)
    rgb = np.flipud(rgb)
    return rgb


def load_preset(preset_name):
    preset_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
    if not os.path.exists(preset_path):
        return parse_positions()
    with open(preset_path) as f:
        data = json.load(f)
    for name in data:
        set_position(SCENE_PATH, name, data[name])
    return parse_positions()


def save_preset(preset_name, positions_dict):
    os.makedirs(PRESETS_DIR, exist_ok=True)
    preset_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
    with open(preset_path, "w") as f:
        json.dump(positions_dict, f, indent=2)
    return f"Saved as '{preset_name}'"


def save_to_xml(positions_dict):
    for name, pos in positions_dict.items():
        set_position(SCENE_PATH, name, pos)
    return "Scene saved to scene.xml"


def list_presets():
    os.makedirs(PRESETS_DIR, exist_ok=True)
    files = [f.replace(".json", "") for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]
    return files


def build_ui():
    positions = parse_positions()
    all_bodies = BALL_NAMES + OBSTACLE_NAMES
    slider_components = {}

    with gr.Blocks(title="Stardance Scene Editor") as ui:
        gr.Markdown("# Stardance Scene Editor")
        gr.Markdown("Adjust ball and obstacle positions. Preview, save, or load presets.")

        with gr.Row():
            with gr.Column(scale=1):
                for name in all_bodies:
                    with gr.Group():
                        gr.Markdown(f"### {name}")
                        pos = positions.get(name, [0.0, 0.0, 0.1])
                        sx = gr.Slider(-1.0, 1.0, value=pos[0], step=0.01, label="X")
                        sy = gr.Slider(-1.0, 1.0, value=pos[1], step=0.01, label="Y")
                        sz = gr.Slider(-0.5, 1.0, value=pos[2], step=0.01, label="Z")
                        slider_components[name] = (sx, sy, sz)

                with gr.Row():
                    btn_preview = gr.Button("Preview", variant="primary")
                    btn_save_xml = gr.Button("Save to scene.xml")

                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=list_presets(),
                        label="Load Preset",
                        interactive=True,
                    )
                    preset_name = gr.Textbox(label="New Preset Name", placeholder="my_scene")
                    btn_save_preset = gr.Button("Save Preset")
                    preset_status = gr.Textbox(label="Status")

            with gr.Column(scale=1):
                preview_img = gr.Image(label="Scene Preview", height=480)

        def get_all_positions(*slider_values):
            result = {}
            idx = 0
            for name in all_bodies:
                result[name] = [
                    slider_values[idx],
                    slider_values[idx + 1],
                    slider_values[idx + 2],
                ]
                idx += 3
            return result

        def on_preview(*slider_values):
            positions_dict = get_all_positions(*slider_values)
            for name, pos in positions_dict.items():
                set_position(SCENE_PATH, name, pos)
            img = render_preview(SCENE_PATH)
            for name in all_bodies:
                pos = positions.get(name, [0.0, 0.0, 0.1])
                set_position(SCENE_PATH, name, pos)
            return img

        def on_save_xml(*slider_values):
            positions_dict = get_all_positions(*slider_values)
            for name, pos in positions_dict.items():
                set_position(SCENE_PATH, name, pos)
            return "Scene saved to scene.xml"

        def on_save_preset(p_name, *slider_values):
            if not p_name:
                return "Enter a preset name"
            positions_dict = get_all_positions(*slider_values)
            result = save_preset(p_name, positions_dict)
            return gr.Dropdown(choices=list_presets(), value=p_name), result

        def on_load_preset(p_name):
            if not p_name:
                return [gr.update()] * (len(all_bodies) * 3)
            data = parse_positions()
            actual = load_preset(p_name)
            updates = []
            for name in all_bodies:
                pos = actual.get(name, data.get(name, [0.0, 0.0, 0.1]))
                updates.append(gr.update(value=pos[0]))
                updates.append(gr.update(value=pos[1]))
                updates.append(gr.update(value=pos[2]))
            return updates

        all_sliders = []
        for name in all_bodies:
            all_sliders.extend(slider_components[name])

        btn_preview.click(on_preview, inputs=all_sliders, outputs=preview_img)
        btn_save_xml.click(on_save_xml, inputs=all_sliders, outputs=preset_status)
        btn_save_preset.click(
            on_save_preset,
            inputs=[preset_name] + all_sliders,
            outputs=[preset_dropdown, preset_status],
        )
        preset_dropdown.change(on_load_preset, inputs=preset_dropdown, outputs=all_sliders)

    return ui


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7861, css="footer {display:none}")
