"""Small helpers for consistent, accessible status presentation."""


def set_status(label, text, state="neutral"):
    label.setText(text)
    if label.property("statusState") == state:
        return
    label.setProperty("statusState", state)
    style = label.style()
    style.unpolish(label)
    style.polish(label)
    label.update()
