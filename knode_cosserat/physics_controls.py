import numpy as np

def calc_controls(control_type, control_arg, del_t, train_len):
    np.random.seed(int(control_arg)) # For random trajectory
    controls = []
    for i in range(1, train_len + 1):
        if control_type == 'sine':
            sin_period = control_arg / del_t # 2.5 seconds was the default
            phase_diff = 2*np.pi/4
            T1 = 6 + np.sin(2*np.pi*i/sin_period + 0*phase_diff)
            T2 = 6 + np.sin(2*np.pi*i/sin_period + 1*phase_diff)
            T3 = 6 + np.sin(2*np.pi*i/sin_period + 2*phase_diff)
            T4 = 6 + np.sin(2*np.pi*i/sin_period + 3*phase_diff)
        elif control_type == 'step':
            step_tension = 0 if i * del_t < 1.5 else control_arg
            T1 = 5 + step_tension
            T2 = 5
            T3 = 5
            T4 = 5 + step_tension
        elif control_type == 'random':
            T1 = 5 + 5*np.random.rand()
            T2 = 5 + 5*np.random.rand()
            T3 = 5 + 5*np.random.rand()
            T4 = 5 + 5*np.random.rand()
        elif control_type == 'ramp':
            T1 = 5 + i*(ramp_speed * del_t)
            T2 = 5
            T3 = 5
            T4 = 5 + i*(ramp_speed * del_t)
        else:
            raise Exception('Unknown control type ' + control_type)
        controls.append([T1, T2, T3, T4])
    return controls
