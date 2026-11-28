"""The Device section, shown on the Setup and Insert Metadata tabs (both show the same choice).

Optional: by default each detection gets what its photo says (nothing if it says nothing).
A person can choose another device type, or customize one (attractors, targets, sensors,
power sources, description), like Classify's Devices editor.

``device_state`` holds ``{"definition": dict | None}`` (None: use what the photos say).
``photos_state`` holds ``{"text": markdown, "found": {label: definition}}`` from the photos.
"""

from types import SimpleNamespace

import gradio as gr

from ui import device_choice as DC


def _table(label, headers):
    return gr.Dataframe(
        label=label,
        headers=headers,
        datatype=["str"] * len(headers),
        type="array",
        row_count=(1, "dynamic"),
        col_count=(len(headers), "fixed"),
        interactive=True,
        wrap=True,
    )


# The power source's number fields; each kind shows its own (DC.POWER_NUMBERS).
POWER_NUMBER_FIELDS = [
    ("capacityWh", "Capacity (Wh)"), ("nominalVoltageV", "Nominal voltage (V)"),
    ("voltageAt80PercentV", "Voltage at 80% full (V)"), ("voltageAt20PercentV", "Voltage at 20% full (V)"),
    ("supplyVoltageV", "Supply voltage (V)"), ("panelPowerW", "Panel power (W)"),
]


def _power_updates(setup):
    """Updates for [kind, setup, name, model, notes, *numbers] showing one power setup (or none)."""
    kind = (setup or {}).get("powerSourceType")
    if kind not in DC.POWER_NUMBERS:
        return [gr.update(value=DC.POWER_NOT_SPECIFIED), gr.update(visible=False, choices=[], value=None),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""), gr.update(visible=False, value="")] + \
               [gr.update(visible=False, value=None) for _ in POWER_NUMBER_FIELDS]
    return [gr.update(value=kind),
            gr.update(visible=True, choices=DC.power_setup_labels(kind, setup), value=setup.get("name")),
            gr.update(visible=True, value=setup.get("name") or ""),
            gr.update(visible=True, value=setup.get("model") or ""),
            gr.update(visible=True, value=setup.get("notes") or "")] + \
           [gr.update(visible=key in DC.POWER_NUMBERS[kind], value=setup.get(key)) for key, _ in POWER_NUMBER_FIELDS]


def build_device_panel(device_state, photos_state, selected_paths):
    """Builds the section in the current layout; wire the tabs together with ``link_device_panels``."""
    base_state = gr.State(None)

    def choose(label, photos, state):
        current = (state or {}).get("definition")
        return {"definition": DC.chosen_definition(label, (photos or {}).get("found"), current=current)}

    # Heading outside the box, like "Additional Processing Files" on the Setup tab.
    gr.Markdown(
        "### Device (optional):\n"
        "Leave this alone to use what each photo says (current Mothbox firmware records the device "
        "in every photo; photos that don't say get no device type). Or choose the device that took "
        "these photos, or customize one."
    )
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=4, min_width=280):
                # Rebuilt whenever the choice or the photos change, so every tab shows the same
                # choice (updates sent to a tab that isn't drawn yet can be lost).
                @gr.render(inputs=[device_state, photos_state])
                def _device_dropdown(state, photos):
                    found = (photos or {}).get("found")
                    current = (state or {}).get("definition")
                    dropdown = gr.Dropdown(
                        label="Device type",
                        choices=DC.choice_labels(found, current=current),
                        value=DC.label_for(current, found),
                        interactive=True,
                        allow_custom_value=False,
                    )
                    dropdown.input(fn=choose, inputs=[dropdown, photos_state, device_state], outputs=[device_state])

            customize_btn = gr.Button("Customize…", size="sm", scale=0, min_width=130)
            check_btn = gr.Button("Check photos again", size="sm", scale=0, min_width=160)
        summary = gr.Markdown(DC.summary(None, None))

    with gr.Accordion("Custom device", open=True, visible=False) as editor:
        gr.Markdown(DC.editor_hint())
        with gr.Row():
            name = gr.Text(label="Name", scale=2)
            maker = gr.Text(label="Maker")
            computer = gr.Text(label="Computer")
        with gr.Row():
            hardware = gr.Text(label="Hardware version", placeholder="e.g. 5.0.6")
            firmware = gr.Text(label="Firmware version", placeholder="e.g. 6.1.2")
        attractors = _table("Attractors (zero or more)", DC.ATTRACTOR_COLUMNS)
        targets = _table("Targets (zero or more)", DC.TARGET_COLUMNS)
        sensors = _table("Sensors (one or more)", DC.SENSOR_COLUMNS)
        gr.Markdown("**Power source** (optional): one battery, grid or solar setup — a standard one, "
                    "one you saved before, or change its details or name to save your own.")
        with gr.Row():
            power_kind = gr.Dropdown(label="Power source", choices=[DC.POWER_NOT_SPECIFIED, *DC.POWER_NUMBERS],
                                     value=DC.POWER_NOT_SPECIFIED, interactive=True)
            power_setup = gr.Dropdown(label="Setup", choices=[], value=None, interactive=True, visible=False)
            power_name = gr.Text(label="Setup name", visible=False,
                                 info="Change the details or the name to save your own setup; it's offered next time.")
            power_model = gr.Text(label="Model", visible=False)
            power_notes = gr.Text(label="Notes", visible=False)
        with gr.Row():
            power_numbers = [gr.Number(label=label, value=None, minimum=0, visible=False) for _, label in POWER_NUMBER_FIELDS]
        description = gr.Textbox(label="Description", placeholder="Notes about the whole device", lines=3)
        problems = gr.Markdown("")
        with gr.Row():
            use_btn = gr.Button("Use this custom device", variant="primary")
            cancel_btn = gr.Button("Cancel", variant="secondary")

    fields = [name, maker, hardware, firmware, computer, description, attractors, targets, sensors]
    power_fields = [power_kind, power_setup, power_name, power_model, power_notes, *power_numbers]

    def choose_power_kind(kind, base):
        current = (base or {}).get("powerSource")
        if current and current.get("powerSourceType") == kind:
            return _power_updates(current)
        first = next(iter(DC.power_setups(kind)), None) if kind in DC.POWER_NUMBERS else None
        return _power_updates(first or ({"powerSourceType": kind, "powerSourceID": f"custom-{kind}", "name": kind}
                                        if kind in DC.POWER_NUMBERS else None))

    def choose_power_setup(label, kind):
        std = next((s for s in DC.power_setups(kind) if s.get("name") == label), None)
        return _power_updates(std) if std else [gr.update() for _ in power_fields]

    def start_editing(state, photos):
        found = (photos or {}).get("found") or {}
        base = (state or {}).get("definition") or next(iter(found.values()), None) or DC.standard_choices().get("Standard Mothbox Pro")
        values = DC.to_editor(base)
        if base and base.get("deviceTypeID") in {d["deviceTypeID"] for d in DC.standard_choices().values()}:
            values["name"] = f"{(base.get('deviceType') or 'Device').removeprefix('Standard ')} (custom)"
        return (
            gr.update(visible=True, open=True),
            base,
            values["name"], values["maker"], values["hardwareVersion"], values["firmwareVersion"],
            values["computer"], values["description"],
            values["attractors"] or [[""] * len(DC.ATTRACTOR_COLUMNS)],
            values["targets"] or [[""] * len(DC.TARGET_COLUMNS)],
            values["sensors"] or [["camera"] + [""] * (len(DC.SENSOR_COLUMNS) - 1)],
            *_power_updates(values["power"] or None),
            "",
        )

    def apply(base, state, *values):
        editor_values = values[:len(fields)]
        kind, setup, setup_name, model, notes, *numbers = values[len(fields):]
        power = {"kind": kind, "setup": setup, "name": setup_name, "model": model, "notes": notes,
                 "numbers": {key: v for (key, _), v in zip(POWER_NUMBER_FIELDS, numbers)}}
        definition, issues = DC.from_editor(base, *editor_values, power=power)
        if issues:
            return state, "\n".join(f"⚠️ {p}" for p in issues), gr.update()
        DC.save_custom_type(definition)
        DC.save_power_setup(definition.get("powerSource"))  # a custom setup is offered on its own next time
        return {"definition": definition}, "", gr.update(visible=False)

    def scan(selected):
        text, found = DC.scan_photos(selected)
        return {"text": text, "found": found}

    customize_btn.click(
        fn=start_editing,
        inputs=[device_state, photos_state],
        outputs=[editor, base_state, *fields, *power_fields, problems],
    )
    power_kind.input(fn=choose_power_kind, inputs=[power_kind, base_state], outputs=power_fields)
    power_setup.input(fn=choose_power_setup, inputs=[power_setup, power_kind], outputs=power_fields)
    use_btn.click(fn=apply, inputs=[base_state, device_state, *fields, *power_fields], outputs=[device_state, problems, editor])
    cancel_btn.click(fn=lambda: gr.update(visible=False), outputs=[editor])
    check_btn.click(fn=scan, inputs=[selected_paths], outputs=[photos_state])

    return SimpleNamespace(summary=summary, editor=editor, scan=scan)


def link_device_panels(panels, device_state, photos_state, selected_paths):
    """Keeps every panel's summary up to date, and reads the photos when the folders change.
    (The dropdowns rebuild themselves from ``device_state`` and ``photos_state``.)"""

    def refresh(state, photos):
        return [DC.summary(state, photos) for _ in panels]

    outputs = [p.summary for p in panels]
    device_state.change(fn=refresh, inputs=[device_state, photos_state], outputs=outputs)
    photos_state.change(fn=refresh, inputs=[device_state, photos_state], outputs=outputs)
    selected_paths.change(fn=panels[0].scan, inputs=[selected_paths], outputs=[photos_state])
