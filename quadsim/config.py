import numpy as np
import yaml
from dataclasses import asdict, dataclass, field, is_dataclass, fields
from copy import deepcopy


# Define a custom representer for NumPy arrays to convert them to lists
def numpy_representer(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data.tolist(), flow_style=False
    )


def dataclass_to_dict(instance):
    if not is_dataclass(instance):
        return instance
    return {k: dataclass_to_dict(v) for k, v in asdict(instance).items()}


def generate_schema(dataclass_type):
    schema = {}
    for field in fields(dataclass_type):
        field_name = field.name
        field_type = field.type
        if is_dataclass(field_type):
            schema[field_name] = generate_schema(field_type)
        else:
            schema[field_name] = field_name
    return schema


def flat_to_nested_dict(flat_dict, schema):
    nested_dict = {}
    for key, sub_keys in schema.items():
        nested_dict[key] = {
            key: flat_dict[key] for key in sub_keys.keys() if key in flat_dict
        }
    return nested_dict


def numpy_to_list(d):
    if isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, dict):
        return {k: numpy_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [numpy_to_list(v) for v in d]
    else:
        return d


def list_to_numpy(d):
    if isinstance(d, list):
        try:
            return np.array(d)
        except:
            return [list_to_numpy(v) for v in d]
    elif isinstance(d, dict):
        return {k: list_to_numpy(v) for k, v in d.items()}
    else:
        return d


def yaml_to_dict(path):
    yaml.add_representer(np.ndarray, numpy_representer)
    with open(path, "r") as file:
        params_loaded = yaml.safe_load(file)
    params = list_to_numpy(params_loaded)
    return params


def dict_to_yaml(params, path):
    yaml.add_representer(np.ndarray, numpy_representer)
    params_converted = numpy_to_list(params)
    with open(path, "w") as file:
        yaml.dump(params_converted, file, default_flow_style=None, sort_keys=False)


def get_affine_scaling_matrices(n, minimum, maximum):
    S = np.diag(np.maximum(np.ones(n), abs(minimum - maximum) / 2))
    c = (maximum + minimum) / 2
    return S, c


@dataclass
class SimConfig:
    model: str
    initial_state: np.ndarray
    initial_control: np.ndarray
    final_state: np.ndarray
    max_state: np.ndarray
    min_state: np.ndarray
    max_control: np.ndarray
    min_control: np.ndarray
    total_time: float
    g: float = -9.81
    m: float = 1.0
    J_b: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    n_states: int = None
    n_controls: int = None
    max_dt: float = 1e2
    min_dt: float = 1e-2
    dt_ss: float = 0.5
    inter_sample: int = 30
    S_t: np.ndarray = None
    c_t: np.ndarray = None
    S_x: np.ndarray = None
    c_x: np.ndarray = None
    S_u: np.ndarray = None
    c_u: np.ndarray = None

    def __post_init__(self):
        self.n_states = self.n_states or 14
        self.n_controls = self.n_controls or 7
        assert (
            self.initial_state.shape[0] == self.n_states - 1
        ), f"Initial state must have {self.n_states - 1} elements"
        assert (
            self.final_state.shape[0] == self.n_states - 1
        ), f"Final state must have {self.n_states - 1} elements"
        assert (
            self.max_state.shape[0] == self.n_states
        ), f"Max state must have {self.n_states} elements"
        assert (
            self.min_state.shape[0] == self.n_states
        ), f"Min state must have {self.n_states} elements"
        assert (
            self.initial_control.shape[0] == self.n_controls
        ), f"Initial control must have {self.n_controls} elements"
        assert (
            self.max_control.shape[0] == self.n_controls
        ), f"Max control must have {self.n_controls} elements"
        assert (
            self.min_control.shape[0] == self.n_controls
        ), f"Min control must have {self.n_controls} elements"

        if self.S_t is None or self.c_t is None:
            self.S_t = max(1, abs(self.min_dt - self.max_dt) / 2)
            self.c_t = (self.max_dt + self.min_dt) / 2
        if self.S_x is None or self.c_x is None:
            self.S_x, self.c_x = get_affine_scaling_matrices(
                self.n_states, self.min_state, self.max_state
            )
        if self.S_u is None or self.c_u is None:
            self.S_u, self.c_u = get_affine_scaling_matrices(
                self.n_controls, self.min_control, self.max_control
            )

@dataclass
class ScpConfig:
    w_tr: float
    lam_fuel: float
    lam_t: float
    lam_vc: float
    lam_vc_ctcs: float
    ep_tr: float
    ep_vb: float
    ep_vc: float
    ep_vc_ctcs: float
    lam_o: float = 0.0
    lam_vp: float = 0.0
    lam_min: float = 0.0
    lam_max: float = 0.0
    k_max: int = 200
    min_fuel: bool = False
    n: int = None
    dis_type: str = "FOH"
    ctcs: bool = True
    uniform_time_grid: bool = False
    fixed_final_time: bool = False
    free_final_drop: int = -1
    min_time_relax: float = 1.0
    w_tr_adapt: float = 1.0
    w_tr_max: float = None
    w_tr_max_scaling_factor: float = None

    def __post_init__(self):
        keys_to_scale = [
            "w_tr",
            "lam_o",
            "lam_fuel",
            "lam_vc",
            "lam_vc_ctcs",
            "lam_t",
            # "w_tr_max",
            "lam_vp",
            "lam_min",
            "lam_max",
        ]
        scale = max(getattr(self, key) for key in keys_to_scale)
        for key in keys_to_scale:
            setattr(self, key, getattr(self, key) / scale)

        if self.w_tr_max_scaling_factor is not None and self.w_tr_max is None:
            self.w_tr_max = self.w_tr_max_scaling_factor * self.w_tr


@dataclass
class ObsConfig:
    n_obs: int = None
    obstacle_centers: list = field(default_factory=list)
    obstacle_axes: list = field(default_factory=list)
    obstacle_radius: list = field(default_factory=list)

    def __post_init__(self):
        if self.n_obs is None:
            self.n_obs = len(self.obstacle_centers)


@dataclass
class VpConfig:
    n_subs: int = 0
    subs_init_pose: np.ndarray = None
    R_sb: np.ndarray = field(default_factory=lambda: np.eye(3))
    tracking: bool = False
    norm: str = "inf"
    alpha_x: float = 4.0
    alpha_y: float = 4.0
    min_range: float = 4.0
    max_range: float = 12.0

    def __post_init__(self):
        if self.n_subs == 0 and self.subs_init_pose is not None and self.subs_init_pose:
            self.n_subs = self.subs_init_pose.shape[0]


@dataclass
class RacingConfig:
    n_gates: int = None
    gate_centers: list = field(default_factory=list)
    gate_nodes: list = field(default_factory=list)

    def __post_init__(self):
        if self.n_gates is None:
            self.n_gates = len(self.gate_nodes)


@dataclass
class WarmConfig:
    warm_start: bool = False
    warm_x_3dof: np.ndarray = None
    warm_u_3dof: np.ndarray = None


@dataclass
class Config:
    sim: SimConfig
    scp: ScpConfig
    obs: ObsConfig
    vp: VpConfig
    racing: RacingConfig
    warm: WarmConfig

    def __post_init__(self):
        pass

    @classmethod
    def from_config(cls, config_instance, savedir=None, savefile="config.yaml"):
        if savedir is not None:
            config_instance.to_yaml(f"{savedir}/{savefile}")

        if config_instance.warm.warm_start:
            cls.generate_double_integrator_config(config_instance, savedir, savefile)

        return config_instance

    @classmethod
    def from_dict(cls, params_in, savedir=None, savefile="config.yaml"):
        # Generate schema for the Config class automatically based on member dataclasses
        schema = generate_schema(cls)
        # Check if input dictionary is flat (no subdictionaries)
        if not any(key in params_in for key in schema.keys()):
            params_in = flat_to_nested_dict(params_in, schema)

        for key in schema.keys():
            if key not in params_in:
                params_in[key] = {}

        # Converting params_in to class instances
        sim_config = SimConfig(**params_in["sim"])
        scp_config = ScpConfig(**params_in["scp"])
        obs_config = ObsConfig(**params_in["obs"])
        vp_config = VpConfig(**params_in["vp"])
        racing_config = RacingConfig(**params_in["racing"])
        warm_config = WarmConfig(**params_in["warm"])

        config_instance = cls(
            sim=sim_config,
            scp=scp_config,
            obs=obs_config,
            vp=vp_config,
            racing=racing_config,
            warm=warm_config,
        )

        config_instance = cls.from_config(config_instance, savedir=savedir, savefile=savefile)

        return config_instance

    @classmethod
    def from_yaml(cls, path, savedir=None, savefile="config.yaml"):
        params_loaded = yaml_to_dict(path)
        return cls.from_dict(params_loaded, savedir=savedir, savefile=savefile)

    def to_yaml(self, path):
        params = asdict(self)
        dict_to_yaml(params, path)

    @classmethod
    def generate_double_integrator_config(
        cls, config_6dof, savedir=None, savefile="config.yaml"
    ):
        sim_6dof = config_6dof.sim
        sim_3dof = deepcopy(sim_6dof)

        # Remove specific fields
        for key in ["n_states", "n_controls", "S_t", "c_t", "S_x", "c_x", "S_u", "c_u"]:
            setattr(sim_3dof, key, None)

        sim_3dof.model = "drone_3dof"
        sim_3dof.initial_state = sim_6dof.initial_state[:6]
        sim_3dof.final_state = sim_6dof.final_state[:6]
        sim_3dof.initial_control = np.delete(sim_6dof.initial_control, [3, 4, 5])
        sim_3dof.max_state = np.delete(sim_6dof.max_state, [6, 7, 8, 9, 10, 11, 12])
        sim_3dof.min_state = np.delete(sim_6dof.min_state, [6, 7, 8, 9, 10, 11, 12])

        acc_max = max(sim_6dof.max_control[:3])
        sim_3dof.max_control = np.array(
            [acc_max, acc_max, acc_max, sim_6dof.max_control[-1]]
        )
        sim_3dof.min_control = np.array(
            [-acc_max, -acc_max, -acc_max, sim_6dof.min_control[-1]]
        )

        sim_3dof.__post_init__()

        if savefile.endswith(".yaml"):
            savefile_modified = savefile.replace(".yaml", "") + "_di.yaml"
        else:
            savefile_modified = savefile + "_di.yaml"

        config_3dof = cls(
            sim=sim_3dof,
            scp=config_6dof.scp,
            obs=config_6dof.obs,
            vp=VpConfig(),
            racing=config_6dof.racing,
            warm=WarmConfig(),
        )

        if savedir is not None:
            config_3dof.to_yaml(f"{savedir}/{savefile_modified}")

        return config_3dof
