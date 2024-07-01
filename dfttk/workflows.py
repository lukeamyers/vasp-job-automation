# Standard library imports
import json
import os
import shutil
import sys

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

# Local application/library specific imports
from custodian.custodian import Custodian
from custodian.vasp.jobs import VaspJob
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer as CMSA

# DFTTK imports
from dfttk.data_extraction import extract_volume
from dfttk.data_extraction import extract_energy

def three_step_relaxation(
    path: str,
    vasp_cmd: list[str],
    handlers: list[str],
    copy_magmom: bool = False,
    backup: bool = False,
    default_settings: bool = True,
    settings_override_2relax: list = None,
    settings_override_3static: list = None,
) -> None:
    """This function runs a three-step relaxation (two consecutive relaxations followed by
       one static) for a given path using VASP. The path should contain the necessary VASP
       input files: POSCAR, POTCAR, INCAR, and KPOINTS.

    Args:
        path: the path to the folder containing the VASP input files
        vasp_cmd: the VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers: custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        copy_magmom: If True, copies the magmom from an OUTCAR file of one run to the INCAR
        file of the next run. Defaults to False.
        backup: If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with
        .orig. Defaults to False.
        default_settings: if True, uses the default settings for the relaxation and static steps.
        settings_override_2relax: a list of settings for the second relaxation step
        settings_override_3static: a list of settings for the static step
    """

    if default_settings:
        settings_override_2relax = [
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
        settings_override_3static = [
            {
                "dict": "INCAR",
                "action": {
                    "_set": {"ALGO": "Normal", "IBRION": -1, "NSW": 0, "ISMEAR": -5}
                },
            },
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
        ]

    original_dir = os.getcwd()
    os.chdir(path)
    step1 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix=".1relax",
        backup=backup,
    )

    step2 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix=".2relax",
        backup=backup,
        settings_override=settings_override_2relax,
    )

    step3 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=True,
        suffix=".3static",
        backup=backup,
        settings_override=settings_override_3static,
    )

    jobs = [step1, step2, step3]
    c = Custodian(handlers, jobs, max_errors=3)
    c.run()
    os.chdir(original_dir)


# TODO: write tests for this function
def ev_curve_series(
    path: str,
    volumes: list[float],
    vasp_cmd: list[str],
    handlers: list[str],
    restarting: bool = False,
    keep_wavecar: bool = False,
    keep_chgcar: bool = False,
    copy_magmom: bool = False,
    default_settings: bool = True,
    settings_override_2relax: list = None,
    settings_override_3static: list = None,
) -> None:
    """This function runs a series of three_step_relaxation calculations for a list of volumes. It starts with the first volume, then
       copies the relevant files to the next volume folder, scales the volume of the POSCAR accordingly, and so on.

    Args:
        path: the path to the folder containing the VASP input files
        volumes: the list of volumes to run the calculations for
        vasp_cmd: the VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers: custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        restarting: for restarting failed jobs. Defaults to False.
        keep_wavecar: if True, does not delete WAVECAR.3static. Defaults to False.
        keep_chgcar: if True, does not delete CHGCAR.3static. Defaults to False.
        copy_magmom: If True, copies the magmom from an OUTCAR file of one run to the INCAR
        file of the next run. Defaults to False.
    """

    # Writes a params.json file to keep track of the parameters used
    errors_subset_list = [handler.errors_subset_to_catch for handler in handlers]
    params = {
        "path": path,
        "volumes": volumes,
        "vasp_cmd": vasp_cmd,
        "handlers": errors_subset_list[0],
        "restarting": restarting,
    }

    n = 0
    params_json_path = os.path.join(path, "params_" + str(n) + ".json")
    while os.path.isfile(params_json_path):
        n += 1
        params_json_path = os.path.join(path, "params_" + str(n) + ".json")

    with open(params_json_path, "w") as file:
        json.dump(params, file)

    # Currently, restarting only supports:
    # 1) Volumes list greater than or equal to the number of vol folders
    # 2) The volumes in the vol folders should match the volumes list in order
    if restarting:

        vol_folders = [
            folder
            for folder in os.listdir(path)
            if os.path.isdir(folder) and folder.startswith("vol")
        ]
        vol_folders = natsorted(vol_folders)

        volumes_started = []
        for vol_folder in vol_folders:
            try:
                volume_started = extract_volume(
                    os.path.join(path, vol_folder, "POSCAR.1relax")
                )
            except Exception as e:
                print(f"possible error: {e}, trying POSCAR")
                try:
                    volume_started = extract_volume(
                        os.path.join(path, vol_folder, "POSCAR")
                    )
                except Exception as e:
                    print(
                        f"Error: {e}. Could not extract volumes from POSCAR files. Do the files POSCAR.1relax or POSCAR exist in each volume folder?"
                    )
                    sys.exit(1)

            volumes_started.append(volume_started)
            rounded_volumes_supplied = [round(volume, 6) for volume in volumes]

        if not volumes_started == rounded_volumes_supplied[: len(volumes_started)]:
            print(
                f"Error: The volumes completed do not match the inputed volumes list of the same number starting from the beginning. \n rounded_input_volumes: {rounded_volumes_supplied} \n volumes_started (rounded): {volumes_started} \n Exiting."
            )
            sys.exit(1)
        else:
            print(
                "The volumes completed match the inputed volumes list of the same number starting from the beginning. Continuing restart"
            )

        j = len(vol_folders) - 1
        last_vol_folder_name = "vol_" + str(j)
        last_vol_folder_path = os.path.join(path, last_vol_folder_name)

        # If the job failed at the third step of three_step_relaxation, restart using the files from the second step
        if all(
            os.path.isfile(os.path.join(last_vol_folder_path, file))
            for file in ["INCAR.2relax", "POSCAR.2relax", "KPOINTS.2relax"]
        ):
            source_name_dest_name = [
                ("INCAR.2relax", "INCAR"),
                ("CONTCAR.2relax", "POSCAR"),
                ("KPOINTS.2relax", "KPOINTS"),
                ("CHGCAR.2relax", "CHGCAR"),
                ("WAVECAR.2relax", "WAVECAR"),
            ]
            for file_name in source_name_dest_name:
                file_source = os.path.join(last_vol_folder_path, file_name[0])
                file_dest = os.path.join(last_vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)

            keep_files = [name[1] for name in source_name_dest_name] + ["POTCAR"]
            for filename in os.listdir(last_vol_folder_path):
                file_path = os.path.join(last_vol_folder_path, filename)
                if filename not in keep_files:
                    os.remove(file_path)

            print("Running three step relaxation for volume " + str(volumes[j]))
            three_step_relaxation(
                last_vol_folder_path,
                vasp_cmd,
                handlers,
                copy_magmom=copy_magmom,
                backup=False,
                default_settings=True,
                settings_override_2relax=settings_override_2relax,
                settings_override_3static=settings_override_3static,
            )

            last_vol_index = j + 1

        # If the job failed at the second step of three_step_relaxation, restart using the files from the first step
        elif all(
            os.path.isfile(os.path.join(last_vol_folder_path, file))
            for file in ["INCAR.1relax", "POSCAR.1relax", "KPOINTS.1relax"]
        ):
            source_name_dest_name = [
                ("INCAR.1relax", "INCAR"),
                ("CONTCAR.1relax", "POSCAR"),
                ("KPOINTS.1relax", "KPOINTS"),
                ("CHGCAR.1relax", "CHGCAR"),
                ("WAVECAR.1relax", "WAVECAR"),
            ]
            for file_name in source_name_dest_name:
                file_source = os.path.join(last_vol_folder_path, file_name[0])
                file_dest = os.path.join(last_vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)

            keep_files = [name[1] for name in source_name_dest_name] + ["POTCAR"]
            for filename in os.listdir(last_vol_folder_path):
                file_path = os.path.join(last_vol_folder_path, filename)
                if filename not in keep_files:
                    os.remove(file_path)

            print("Running three step relaxation for volume " + str(volumes[j]))
            three_step_relaxation(
                last_vol_folder_path,
                vasp_cmd,
                handlers,
                copy_magmom=copy_magmom,
                backup=False,
                default_settings=True,
                settings_override_2relax=settings_override_2relax,
                settings_override_3static=settings_override_3static,
            )
            last_vol_index = j + 1

        # If the job failed at the first step of three_step_relaxation, delete the folder
        else:
            shutil.rmtree(last_vol_folder_path)
            last_vol_index = j

    files_to_delete = [
        "WAVECAR.1relax",
        "WAVECAR.2relax",
        "WAVECAR.3static",
        "CHGCAR.3static",
        "CHGCAR.1relax",
        "CHGCAR.2relax",
        "CHG.1relax",
        "CHG.2relax",
        "CHG.3static",
        "PROCAR.1relax",
        "PROCAR.2relax",
        "PROCAR.3static",
    ]
    if keep_wavecar:
        files_to_delete.remove("WAVECAR.3static")
    if keep_chgcar:
        files_to_delete.remove("CHGCAR.3static")

    # This starts the EV curve calculations in series. If restarting, skip the volumes that have already been completed.
    for i, vol in enumerate(volumes):
        if restarting and i < last_vol_index:
            continue

        vol_folder_name = "vol_" + str(i)
        vol_folder_path = os.path.join(path, vol_folder_name)
        os.makedirs(vol_folder_path)

        if i == 0:
            files_to_copy = ["INCAR", "KPOINTS", "POSCAR", "POTCAR"]
            for file_name in files_to_copy:
                if os.path.isfile(os.path.join(path, file_name)):
                    shutil.copy2(
                        os.path.join(path, file_name),
                        os.path.join(vol_folder_path, file_name),
                    )
        else:
            previous_vol_folder_path = os.path.join(path, "vol_" + str(i - 1))
            source_name_dest_name = [
                ("CONTCAR.3static", "POSCAR"),
                ("INCAR.2relax", "INCAR"),
                ("KPOINTS.1relax", "KPOINTS"),
                ("POTCAR", "POTCAR"),
                ("WAVECAR.3static", "WAVECAR"),
                ("CHGCAR.3static", "CHGCAR"),
            ]
            for file_name in source_name_dest_name:
                file_source = os.path.join(previous_vol_folder_path, file_name[0])
                file_dest = os.path.join(vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)

            paths_to_delete = []

            for file_name in files_to_delete:
                file_path = os.path.join(previous_vol_folder_path, file_name)
                paths_to_delete.append(file_path)

            for file_path in paths_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                elif restarting and i == last_vol_index:
                    pass

        poscar = os.path.join(vol_folder_path, "POSCAR")
        struct = Structure.from_file(poscar)
        struct.scale_lattice(vol)
        struct.to_file(poscar, "POSCAR")

        print("Running three step relaxation for volume " + str(vol))
        three_step_relaxation(
            vol_folder_path,
            vasp_cmd,
            handlers,
            backup=False,
            copy_magmom=copy_magmom,
            default_settings=default_settings,
            settings_override_2relax=settings_override_2relax,
            settings_override_3static=settings_override_3static,
        )

    previous_vol_folder_path = os.path.join(path, "vol_" + str(i))
    paths_to_delete = []
    for file_name in files_to_delete:
        file_path = os.path.join(previous_vol_folder_path, file_name)
        paths_to_delete.append(file_path)

    for file_path in paths_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)


def charge_density_difference(
    path: str, vasp_cmd: list[str], handlers: list[str], backup: bool = False
):
    """
    Runs a charge density difference calculation for a configuration in a subdirectory of the given path.
    called charge_density_difference. The charge density difference is calculated as the difference between
    The charge density of the final electronic step and the charge density of a single step.

    Args:
        path: The path that contains the INCAR, POSCAR KPOINTS, and POTCAR.
        vasp_cmd: The command to run VASP.
        handlers: A list of error handlers that will be used during the calculation.
        Refer to custodian.vasp.handlers
        backup:  Whether to backup the initial input files. If True, the INCAR,
        KPOINTS, POSCAR and POTCAR will be copied with a “.orig” appended. Defaults to True.

    Returns:
        The charge density difference between the final electronic step and
        a single step.
    """
    original_dir = os.getcwd()
    os.chdir(path)
    os.mkdir("charge_density_difference")
    shutil.copy2("POSCAR", "charge_density_difference/POSCAR")
    shutil.copy2("POTCAR", "charge_density_difference/POTCAR")
    shutil.copy2("INCAR", "charge_density_difference/INCAR")
    shutil.copy2("KPOINTS", "charge_density_difference/KPOINTS")
    os.chdir("charge_density_difference")

    reference_job = VaspJob(
        vasp_cmd=vasp_cmd,
        final=False,
        suffix=".reference",
        backup=backup,
        settings_override=[
            {
                "dict": "INCAR",
                "action": {
                    "_set": {
                        "EDIFF": "1E-6",
                        "IBRION": -1,
                        "NSW": 1,
                        "ISIF": 2,
                        "NELM": 1,
                        "ISMEAR": -5,
                        "SIGMA": 0.05,
                        "LCHARG": True,
                    }
                },
            },
        ],
    )

    charge_density_job = VaspJob(
        vasp_cmd=vasp_cmd,
        final=True,
        suffix=".charge_density",
        backup=backup,
        settings_override=[
            {
                "dict": "INCAR",
                "action": {
                    "_set": {
                        "EDIFF": "1E-6",
                        "IBRION": -1,
                        "NSW": 1,
                        "ISIF": 2,
                        "NELM": 100,
                        "ISMEAR": -5,
                        "SIGMA": 0.05,
                        "LCHARG": True,
                    }
                },
            },
        ],
    )

    jobs = [reference_job, charge_density_job]
    c = Custodian(handlers, jobs, max_errors=3)
    c.run()

    final = Chgcar.from_file("CHGCAR.charge_density")
    reference = Chgcar.from_file("CHGCAR.reference")
    difference = final - reference
    difference.write_file("CHGCAR.difference")

    os.chdir(original_dir)

    return difference


def custodian_errors_location(path: str) -> None:
    vol_folders = [d for d in os.listdir(path) if d.startswith("vol")]
    for vol_folder in vol_folders:
        error_folders = [
            f
            for f in os.listdir(os.path.join(path, vol_folder))
            if f.startswith("error")
        ]
        if len(error_folders) > 0:
            print(f"In {vol_folder} there are error folders: {error_folders}")


def NELM_reached(path: str) -> None:
    start_dir = path
    target_line = "The electronic self-consistency was not achieved in the given"
    for dirpath, dirs, files in os.walk(start_dir):
        for filename in files:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, "r", errors="ignore") as file:
                for line in file:
                    if target_line in line:
                        print(f"{filepath} has reached NELM.")
                        break


# TODO: add a way to restart the job if it has failed
# TODO: add a way to override the default settings
def run_phonons(
    vasp_cmd: list[str],
    handlers: list[str],
    copy_magmom: bool = False,
    backup: bool = False,
):

    step1 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix=".1relax",
        backup=backup,
    )

    step2 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=True,
        suffix=".2phonons",
        backup=backup,
        settings_override=[
            {
                "dict": "INCAR",
                "action": {
                    "_set": {
                        "EDIFF": "1E-6",
                        "IBRION": 6,
                        "NSW": 1,
                        "ISIF": 0,
                        "POTIM": 0.015,
                        "ISYM": 2,
                        "NCORE": 1,
                    }
                },
            },
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
        ],
    )

    jobs = [step1, step2]
    c = Custodian(handlers, jobs, max_errors=3)
    c.run()


def phonons_parallel(
    path: str,
    phonon_volumes: list[float],
    supercell_size: list[int],
    kppa: float,
    run_file: str,
) -> None:

    # Create a new run_file to run the phonon calculations
    script_name = sys.argv[0]
    with open(script_name, "r") as file:
        script_contents = file.read()
        script_contents = "\n".join(
            [
                line
                for line in script_contents.split("\n")
                if "workflows.phonons_parallel" not in line
            ]
        )

    with open(run_file, "r") as file:
        run_file_contents = file.read()

    new_run_file = run_file_contents + "\n"
    new_run_file += "\n"
    new_run_file += "python << END_OF_PYTHON\n"
    new_run_file += script_contents
    new_run_file += "workflows.run_phonons(vasp_cmd, handlers)\n"
    new_run_file += "END_OF_PYTHON\n"

    # Copy files to phonon folders
    vol_folders = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder)) and folder.startswith("vol")
    ]

    ev_volumes_finished = []
    ev_folder_names = []
    for vol_folder in vol_folders:
        structure_path = os.path.join(path, vol_folder, "CONTCAR.3static")
        ev_volumes_finished.append(extract_volume(structure_path))
        ev_folder_names.append(vol_folder)

    ev_volumes_and_folders_finished = [
        [a, b] for a, b in zip(ev_volumes_finished, ev_folder_names)
    ]

    for i in range(len(ev_volumes_and_folders_finished)):
        ev_volumes_and_folders_finished[i][1] = ev_volumes_and_folders_finished[i][
            1
        ].replace("vol_", "")

    phonon_volumes_and_folders = []
    for ev_volume_finished, folder in ev_volumes_and_folders_finished:
        if ev_volume_finished in phonon_volumes:
            phonon_volumes_and_folders.append([ev_volume_finished, folder])

    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        os.makedirs(os.path.join(path, f"phonon_{phonon_folder}"), exist_ok=True)

    source_name_dest_name = [
        ("CONTCAR.3static", "POSCAR"),
        ("INCAR.2relax", "INCAR"),
        ("POTCAR", "POTCAR"),
        ("WAVECAR.3static", "WAVECAR"),
    ]

    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        for source_name, dest_name in source_name_dest_name:
            file_source = os.path.join(path, f"vol_{phonon_folder}", source_name)
            file_dest = os.path.join(path, f"phonon_{phonon_folder}", dest_name)
            if os.path.isfile(file_source):
                shutil.copy2(file_source, file_dest)

    # Create a supercell and write the KPOINTS file
    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        structure = Structure.from_file(
            os.path.join(path, f"phonon_{phonon_folder}", "POSCAR")
        )
        structure.make_supercell(supercell_size)
        structure.to_file(
            os.path.join(path, f"phonon_{phonon_folder}", "POSCAR"), "POSCAR"
        )
        kpoints = Kpoints.automatic_density(structure, kppa, force_gamma=True)
        kpoints.write_file(os.path.join(path, f"phonon_{phonon_folder}", "KPOINTS"))

    # Run the phonon calculations in parallel
    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        os.chdir(os.path.join(path, f"phonon_{phonon_folder}"))
        with open("run_phonons", "w") as file:
            file.write(new_run_file)
        os.system("sbatch run_phonons")
        os.chdir(path)


def process_phonon_dos_YPHON(path: str):

    # Go to each phonon folder and copy the CONTCAR, OUTCAR, and vasprun.xml files to the phonon_dos folder to be processed by YPHON
    phonon_folders = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder)) and folder.startswith("phonon")
    ]

    for phonon_folder in phonon_folders:
        os.chdir(os.path.join(path, phonon_folder))
        if not os.path.exists("phonon_dos"):
            os.makedirs("phonon_dos", exist_ok=True)
        shutil.copy(
            os.path.join(path, phonon_folder, "CONTCAR.2phonons"),
            os.path.join(path, phonon_folder, "phonon_dos", "CONTCAR"),
        )
        shutil.copy(
            os.path.join(path, phonon_folder, "OUTCAR.2phonons"),
            os.path.join(path, phonon_folder, "phonon_dos", "OUTCAR"),
        )
        shutil.copy(
            os.path.join(path, phonon_folder, "vasprun.xml.2phonons"),
            os.path.join(path, phonon_folder, "phonon_dos", "vasprun.xml"),
        )

        os.chdir(os.path.join(path, phonon_folder, "phonon_dos"))
        index = phonon_folder.split("_")[1]
        structure = Structure.from_file("CONTCAR")
        number_of_atoms = structure.num_sites
        volume = extract_volume("CONTCAR")
        volume_per_atom = volume / number_of_atoms

        with open("volph_" + index, "w") as f:
            f.write(str(volume_per_atom))

        # YPHON commands
        os.system("vasp_fij")
        os.system("Yphon <superfij.out")

        os.rename("vdos.out", "vdos_" + index)
        os.chdir(path)

    os.makedirs("YPHON_results", exist_ok=True)
    for phonon_folder in phonon_folders:
        shutil.copy(
            os.path.join(
                path, phonon_folder, "phonon_dos", "vdos_" + phonon_folder.split("_")[1]
            ),
            os.path.join(path, "YPHON_results", "vdos_" + phonon_folder.split("_")[1]),
        )
        shutil.copy(
            os.path.join(
                path,
                phonon_folder,
                "phonon_dos",
                "volph_" + phonon_folder.split("_")[1],
            ),
            os.path.join(path, "YPHON_results", "volph_" + phonon_folder.split("_")[1]),
        )


def kpoints_conv_test(
    path: str,
    kppa_list: list[float],
    vasp_cmd: list[str],
    handlers: list[str],
    force_gamma: bool = True,
    backup: bool = False,
):
    """This function runs a series of VASP calculations with different k-point densities for convergence testing.

    Args:
        path: the path to the folder containing the VASP input files
        kppa_list: the list of k-point densities to run the calculations for
        vasp_cmd: the VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers: custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        force_gamma: If True, forces a gamma-centered mesh. Defaults to True.
        backup: If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with
        .orig. Defaults to False.
    """

    original_dir = os.getcwd()
    kpoints_conv_dir = os.path.join(path, "kpoints_conv")
    os.makedirs(kpoints_conv_dir)

    shutil.copy2(os.path.join(path, "POSCAR"), os.path.join(kpoints_conv_dir, "POSCAR"))
    shutil.copy2(os.path.join(path, "POTCAR"), os.path.join(kpoints_conv_dir, "POTCAR"))
    shutil.copy2(os.path.join(path, "INCAR"), os.path.join(kpoints_conv_dir, "INCAR"))

    os.chdir(kpoints_conv_dir)
    struct = Structure.from_file("POSCAR")
    for i, kppa in enumerate(kppa_list):
        kpoints = Kpoints.automatic_density(struct, kppa, force_gamma=force_gamma)
        kpoints.write_file("KPOINTS")

        if i == len(kppa_list) - 1:
            final = True
        else:
            final = False

        job = VaspJob(
            vasp_cmd=vasp_cmd,
            final=final,
            backup=backup,
            suffix=f".{kppa}",
            settings_override=[
                {"dict": "INCAR", "action": {"_set": {"IBRION": -1, "NSW": 0}}}
            ],
        )
        c = Custodian(handlers, [job], max_errors=3)
        c.run()

        if os.path.isfile(f"WAVECAR.{kppa}"):
            os.remove(f"WAVECAR.{kppa}")
        if os.path.isfile(f"CHGCAR.{kppa}"):
            os.remove(f"CHGCAR.{kppa}")
        if os.path.isfile(f"CHG.{kppa}"):
            os.remove(f"CHG.{kppa}")
        if os.path.isfile(f"PROCAR.{kppa}"):
            os.remove(f"PROCAR.{kppa}")
    os.chdir(original_dir)
    return None


def calculate_kpoint_conv(path: str, kppa_list: list[str], plot: bool = True):
    """This function calculates the energy convergence with respect to k-point density and plots the results.

    Args:
        path: the path to the folder containing the VASP input files
        kppa_list: the list of k-point densities to run the calculations for
        plot: If True, plots the results. Defaults to True.
    """
    original_dir = os.getcwd()
    kpoints_conv_dir = os.path.join(path, "kpoints_conv")

    os.chdir(kpoints_conv_dir)
    data = []
    for kppa in kppa_list:
        energy = extract_energy(f"OSZICAR.{kppa}")
        data.append([kppa, energy])
    data = np.array(data)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    num_atoms = len(Structure.from_file(f"POSCAR.{kppa}").sites)
    sorted_data = np.column_stack((sorted_data, np.zeros(len(sorted_data))))
    sorted_data[1:, 2] = (sorted_data[1:, 1] - sorted_data[:-1, 1]) / num_atoms * 1000
    os.chdir(path)
    np.savetxt("kppa_energy.txt", sorted_data, fmt="%f")

    if plot:
        fig, axis = plt.subplots(1, 2, figsize=(12, 6))
        axis[0].plot(sorted_data[:, 0], sorted_data[:, 1], marker="o")
        axis[0].set_xlabel("k-point density")
        axis[0].set_ylabel("Energy (eV)")
        axis[1].plot(sorted_data[:, 0], sorted_data[:, 2], marker="o")
        axis[1].axhline(y=1, color="black", linestyle="--")
        axis[1].axhline(y=-1, color="black", linestyle="--")
        axis[1].set_xlabel("k-point density")
        axis[1].set_ylabel("ΔEnergy (meV/atom)")
        plt.tight_layout()
        plt.savefig("kpoint_conv.png", dpi=300)
    os.chdir(original_dir)


def encut_conv_test(
    path: str,
    encut_list: list[float],
    vasp_cmd: list[str],
    handlers: list[str],
    backup: bool = False,
):
    """This function runs a series of VASP calculations with different ENCUT values for convergence testing.

    Args:
        path: the path to the folder containing the VASP input files
        encut_list: the list of ENCUT values to run the calculations for
        vasp_cmd: the VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers: custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        backup: If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with
        .orig. Defaults to False.
    """
    original_dir = os.getcwd()
    encut_conv_dir = os.path.join(path, "encut_conv")
    os.makedirs(encut_conv_dir)

    shutil.copy2(os.path.join(path, "POSCAR"), os.path.join(encut_conv_dir, "POSCAR"))
    shutil.copy2(os.path.join(path, "KPOINTS"), os.path.join(encut_conv_dir, "KPOINTS"))
    shutil.copy2(os.path.join(path, "POTCAR"), os.path.join(encut_conv_dir, "POTCAR"))
    shutil.copy2(os.path.join(path, "INCAR"), os.path.join(encut_conv_dir, "INCAR"))

    os.chdir(encut_conv_dir)
    for i, encut in enumerate(encut_list):
        if i == len(encut_list) - 1:
            final = True
        else:
            final = False

        job = VaspJob(
            vasp_cmd=vasp_cmd,
            final=final,
            backup=backup,
            suffix=f".{encut}",
            settings_override=[
                {
                    "dict": "INCAR",
                    "action": {"_set": {"IBRION": -1, "NSW": 0, "ENCUT": encut}},
                }
            ],
        )
        c = Custodian(handlers, [job], max_errors=3)
        c.run()

        if os.path.isfile(f"WAVECAR.{encut}"):
            os.remove(f"WAVECAR.{encut}")
        if os.path.isfile(f"CHGCAR.{encut}"):
            os.remove(f"CHGCAR.{encut}")
        if os.path.isfile(f"CHG.{encut}"):
            os.remove(f"CHG.{encut}")
        if os.path.isfile(f"PROCAR.{encut}"):
            os.remove(f"PROCAR.{encut}")
    os.chdir(original_dir)


def calculate_encut_conv(path: str, encut_list: str, plot: bool = True):
    """This function calculates the energy convergence with respect to ENCUT and plots the results.

    Args:
        path: the path to the folder containing the VASP input files
        encut_list: the list of ENCUT values to run the calculations for
        plot: If True, plots the results. Defaults to True.
    """
    original_dir = os.getcwd()
    encut_conv_dir = os.path.join(path, "encut_conv")

    os.chdir(encut_conv_dir)
    data = []
    for encut in encut_list:
        energy = extract_energy(f"OSZICAR.{encut}")
        data.append([encut, energy])
    data = np.array(data)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    num_atoms = len(Structure.from_file(f"POSCAR.{encut}").sites)
    sorted_data = np.column_stack((sorted_data, np.zeros(len(sorted_data))))
    sorted_data[1:, 2] = (sorted_data[1:, 1] - sorted_data[:-1, 1]) / num_atoms * 1000
    os.chdir(path)
    np.savetxt("encut_energy.txt", sorted_data, fmt="%f")

    if plot:
        fig, axis = plt.subplots(1, 2, figsize=(12, 6))
        axis[0].plot(sorted_data[:, 0], sorted_data[:, 1], marker="o")
        axis[0].set_xlabel("ENCUT (eV)")
        axis[0].set_ylabel("Energy (eV)")
        axis[1].plot(sorted_data[:, 0], sorted_data[:, 2], marker="o")
        axis[1].axhline(y=1, color="black", linestyle="--")
        axis[1].axhline(y=-1, color="black", linestyle="--")
        axis[1].set_xlabel("ENCUT")
        axis[1].set_ylabel("ΔEnergy (meV/atom)")
        plt.tight_layout()
        plt.savefig("encut_conv.png", dpi=300)
    os.chdir(original_dir)
