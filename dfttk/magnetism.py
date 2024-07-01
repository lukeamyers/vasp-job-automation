# Standard library imports
import os
import sys
import itertools
import numbers
import shutil
import subprocess



# Related third party imports
import numpy as np
import pandas as pd
from natsort import natsorted


# Local application/library specific imports
from ase import io
from pymatgen.core.structure import Structure
from pymatgen.analysis.magnetism.analyzer import \
    CollinearMagneticStructureAnalyzer as CMSA
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Poscar


# DFTTK imports
from dfttk.data_extraction import (
    extract_tot_mag_data, extract_input_mag_data
)


def determine_magnetic_ordering(
    df: pd.DataFrame,
    magmom_tolerance: float = 1e-12,
    total_magnetic_moment_tolerance: float = 1e-12
) -> str:
    """Determines the magnetic ordering of a structure from the magnetization 
    data in a pandas DataFrame. e.g. 'FM', 'AFM', 'FiM', 'NM', 'SF'

    Args:
        df (pandas DataFrame): a pandas DataFrame containing the magnetization data
        magmom_tolerance (float, optional): the tolerance for the total magnetic moment on each atom to be considered zero.
        Total_magmom_tolerance (float, optional) the tolerance for the sum of the total magnentic moments for each atom.
        Defaults to 1e-12 to handle floating point errors.

    Returns:
        The magnetic ordering of the structure
    """

    if (np.isclose(df["tot"], 0, atol=magmom_tolerance)).all():
        return "NM"
    elif np.isclose(df["tot"].sum(), 0, atol=total_magnetic_moment_tolerance) == True:
        return "AFM"
    elif (df["tot"] >= 0 + magmom_tolerance).all() or (
        df["tot"] <= 0 - magmom_tolerance
    ).all():
        return "FM"
    elif (df["tot"] > 0 + magmom_tolerance).sum() == (
        df["tot"] < 0 - magmom_tolerance
    ).sum():
        return "FiM"
    else:
        return "SF"

def get_magnetic_structure(poscar: str, outcar: str) -> Structure:
    """Combines the magmom data from the outcar with the structure from the poscar
    to return a pymatgen magnetic Structures object (e.g. Structures with
    associated magmom tags).

    Args:
        poscar (str): name of the POSCAR file
        outcar (str): name of the OUTCAR file

    Returns:
        Structure: pymatgen Structure object with magmom tags
    """
    structure = Structure.from_file(poscar)
    mag_data = extract_tot_mag_data(outcar)
    structure.add_site_property("magmom", mag_data["tot"])
    return structure

#TODO: make this magnetic/non-magnetic agnostic
def equivalent_orderings(path: str,
                         contcar_name: str ='CONTCAR',
                         outcar_name: str = 'OUTCAR'
) -> bool:
    """finds equivalent magnetic orderings for a set of configurations in a path
    Works rather slow. Needs to be optimized. 350 configurations takes about 10 minutes.

    Args:
        path: Path to "configurations" folder
        contcar_name: name of the CONTCAR file. Defaults to 'CONTCAR'.
        outcar_name: name of the OUTCAR file. Defaults to 'OUTCAR'.

    Raises:
        FileNotFoundError: if the contcar/outcar files are not found for a config

    Returns:
        a dictionary where the keys are the configurations and the values are lists of configurations with matching magnetic ordering
    """    
    struct_dict = {}
    for config_dir in os.listdir(path):
        config_dir_path = os.path.join(path, config_dir)
        if os.path.isdir(config_dir_path) and config_dir.startswith("config_"):
            for subdir in os.listdir(config_dir_path):
                subdir_path = os.path.join(config_dir_path, subdir)
                if os.path.isdir(subdir_path) and subdir.startswith("vol_"):
                    try:
                        magnetic_structure = get_magnetic_structure(
                            os.path.join(subdir_path, contcar_name),
                            os.path.join(subdir_path, outcar_name)
                        )
                        config = config_dir.split("config_")[1]
                        struct_dict[config] = magnetic_structure
                        structure_found = True
                        break
                    except FileNotFoundError as e:
                        print(f"missing CONTCAR/OUTCAR in {subdir_path}: {e}. Did you use the correct CONTCAR/OUTCAR name?")
            if not structure_found:
                raise FileNotFoundError(f"Could not make magnetic structure for config in {config_dir_path}")    
    equivalence_dict = {config: [] for config in struct_dict.keys()}
    items = struct_dict.items()
    for i, (config, magnetic_structure) in enumerate(items):
        analyzer = CMSA(magnetic_structure)
        for remaining_config, remaining_magnetic_structure in itertools.islice(
            items,
            i+1,
            len(struct_dict)
            ):
            if analyzer.matches_ordering(remaining_magnetic_structure):
                equivalence_dict[config].append(remaining_config)
                equivalence_dict[remaining_config].append(config)
    return equivalence_dict

def remove_equivalent_orderings(
    df: pd.DataFrame,
    equivalence_dict: dict
) -> pd.DataFrame:
    remove_list = []
    sorted_df = df.sort_values(by='energy_per_atom')
    for index, row in sorted_df.iterrows():
        if row['config'] in remove_list:
            continue
        elif equivalence_dict[row['config']] == []:
            continue
        else:
            remove_list.extend(equivalence_dict[row['config']])
    
    #keep rows that are not in the remove_list
    return df[~df['config'].isin(remove_list)]

#TODO: support specify min and max for each ion (dict) and min/max (tuple) for
# magmom_tol. it may be beneficial to have a range of acceptable values instead
# a tolerance.
def significant_magmom_change(
    outcar_path: str = "OUTCAR",
    magmom_tol: float = 0.5
) -> bool:
    """determines if the resulting magnetic moment is significantly different from the input magnetic moment for any of the atoms.

    Args:
        outcar_path: Path to the OUTCAR. Defaults to "OUTCAR".
        magmom_tol: tolerance for change in magnetic moment for each atom. Defaults to 0.5.

    Raises:
        ValueError: if the magmom_tol is not a real number (float, int, etc).

    Returns:
        bool: True if at least one of the atoms in the struct has a resulting magnetic moment that is significantly different from the input.
    """    
    input_magmoms = extract_input_mag_data(outcar_path)
    output_magmoms = extract_tot_mag_data(outcar_path)
    
    if isinstance(magmom_tol, numbers.Real):
        magmom_tol = abs(magmom_tol)
        min_df = input_magmoms.copy()
        max_df = input_magmoms.copy()
        min_df['tot'] = min_df['tot'] - magmom_tol
        max_df['tot'] = max_df['tot'] + magmom_tol
    # elif isinstance(magmom_tol, dict):
    #     pass
    # elif isinstance(magmom_tol, tuple):
    #     pass
    else:
        raise ValueError("magmom_tol must be a real number (float, int, etc) or a dictionary")
    for index, row in output_magmoms.iterrows():
        if row['tot'] < min_df['tot'][index] or row['tot'] > max_df['tot'][index]:
            return True
    return False
    


"""
TODO: Clean all functions in this section. Some may not be needed, may be
rewritten, or belong elsewhere. Some functions may be improved using pymatgen.
In particular the pyatgens structure enumerator instead of ATAT.
A lot is not worth messing with as it will be replaced by pymatgen stuff.
"""

def write_to_file(filename, lines):
    with open(filename, 'w') as file:
        file.writelines(lines)

def parse(ywoutput, directory='strs'):
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the 'strs' directory if it doesn't exist
    
    with open(ywoutput, 'r') as file:
        lines = file.readlines()

    current_lines = []
    file_number = 1

    for line in lines:
        current_lines.append(line)

        # Check if the line contains the "end" marker
        if 'end' in line:
            output_filename = os.path.join(directory, f'str.{file_number}')
            write_to_file(output_filename, current_lines)
            current_lines = []  # Reset the current_lines list for the next section
            file_number += 1

def convert_strs_to_poscars(directory='strs', configurations_directory='configurations'):
    # Get a list of files in the specified directory
    str_files = [filename for filename in os.listdir(directory) if filename.startswith('str')]

    for structure_from in str_files:
        output_directory = os.path.join(configurations_directory, 'config_' + structure_from[4:])
        structure_to = os.path.join(output_directory, 'POSCAR')

        if not os.path.exists(f'{directory}/{structure_from}'):
            print(f'EEEE ATAT file {structure_from} does not exist in the path!')
            print('EEEE You have to specify the ATAT file in this Python script.')
            print('EEEE exiting ...')
            sys.exit(1)

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Convert str.out to CIF format first
        tmp = 'tmp.cif'
        cmd = f'str2cif < {directory}/{structure_from} > {tmp}'
        os.system(cmd)

        # Then convert CIF to POSCAR using ASE
        atoms = io.read(tmp)
        atoms.write(structure_to, format='vasp')

        # Clean up
        os.remove(tmp)

def count_atoms(strs_dir='strs'):
    kept = []
    data_list = []
    for file_name in os.listdir(strs_dir):
        frequency_dict = {'file_number': file_name[4:]}
        with open(os.path.join(strs_dir, file_name), 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    parts = line.split()
                    if len(parts) == 4:
                        value = parts[3]
                        if value in frequency_dict:
                            frequency_dict[value] += 1
                        else:
                            frequency_dict[value] = 1
        data_list.append(frequency_dict)
    df = pd.DataFrame(data_list) 
    df = df.fillna(0) 
    return df

def remove_spin_up_less_than_down(strs_path, atom_count_df, atom_up, atom_down): #takes the dataframe from count_atoms and removes all files where the number of spin up atoms is less than the number of spin down
    for index, row in atom_count_df.iterrows():
        if row[atom_up] < row[atom_down]:
            file_path = os.path.join(strs_path, f'str.{row["file_number"]}')
            os.remove(file_path)

"""
magmoms is a dictionary that looks like:
magmoms = {'Fe+': 5, 'Fe-': -5, 'O': 0}
"""

def make_incars(magmoms, incar='INCAR', configurations_directory='configurations', strs_dir='strs'): #takes an INCAR file and adds a line for the magnetic moment in accordance with the str file
    str_files = [filename for filename in os.listdir(strs_dir) if filename.startswith('str')]
    for str_file in str_files:
        mag_mom_list = []
        output_directory = os.path.join(configurations_directory, 'config_' + str_file[4:])
        incar_to = os.path.join(output_directory, 'INCAR')
        atoms = list(magmoms.keys())
        with open(os.path.join(strs_dir, str_file), 'r') as file:
            lines = file.readlines()
            for line in lines:
                count = 0
                for atom in atoms:
                    if atom in line:
                        mag_mom_list.append(magmoms[atom])
                        count += 1
                if count not in [0, 1]:
                    raise ValueError(f'Error: {count} atoms found in line {line} in file {str_file}')
        mag_mom_string = ' '.join(mag_mom_list)
        shutil.copy(incar, incar_to)
        # Read the file and store lines in a list, excluding lines starting with 'MAGMOM'
        with open(incar_to, 'r') as file:
            lines = [line for line in file if not line.startswith('MAGMOM')]

            # Append the new 'MAGMOM' line to the list
            lines.append('\nMAGMOM = %s' % mag_mom_string)

        # Write the modified list back to the file
        with open(incar_to, 'w') as file:
           file.writelines(lines)



"""
This function differs from make_incars and convert_strs_to_poscars in that it
requires the config_dirs to already exist. it also assume that there is a
poscar inside that directory.

kppa (float) – Grid density

force_gamma (bool) – Force a gamma centered mesh (default is to use gamma only for hexagonal cells or odd meshes)
configurations_directory (str) – Path to the configurations directory
"""
def make_kpoints(kppa, force_gamma=False, configurations_directory='configurations'):
    config_dirs = [dir for dir in os.listdir(configurations_directory) if os.path.isdir(os.path.join(configurations_directory, dir))]
    for config_dir in config_dirs:
        structure = Structure.from_file(os.path.join(configurations_directory, config_dir, 'POSCAR'))
        kpoints = Kpoints.automatic_density(structure, kppa, force_gamma=force_gamma)
        kpoints.write_file(os.path.join(configurations_directory, config_dir, 'KPOINTS'))


"""
function that creates a submit script for each configuration directory. It
also changes the job name to be the same as the configuration name. that is,
everything after config_ in the directory name.

# TODO: generalize this function (if needed). move it to a more general location.
"""
def create_submit_scripts(configurations_directory='configurations', submit_script='submit.sh'):
    config_dirs = [dir for dir in os.listdir(configurations_directory) if os.path.isdir(os.path.join(configurations_directory, dir))]
    with open(submit_script, 'r') as submit_file:
        lines = submit_file.readlines()
    for config_dir in config_dirs:
        new_job_name = config_dir.split('config_')[-1]
        submit_script_path = os.path.join(configurations_directory, config_dir, submit_script)
        with open(submit_script_path, 'w') as file:
            for line in lines:
                if line.startswith('#SBATCH --job-name='):
                    file.write(f'#SBATCH --job-name={new_job_name}\n')
                else:
                    file.write(line)

"""
this function is a patch to rearrange the sites and magmoms in the POSCAR and
INCAR files. If the sites are not grouped by specie, VASP will look for more
potentials than supplied/necessary.
"""
def rearrange_sites_and_magmoms(config_dir):
    incar_file = os.path.join(config_dir, 'INCAR')
    poscar_file = os.path.join(config_dir, 'POSCAR')
    struct = Structure.from_file(poscar_file) # read poscar
    orig_magmom_df = extract_input_mag_data(incar_file) # read magmom from incar
    orig_magmoms = orig_magmom_df['tot'].tolist() # get the magmoms
    struct.add_site_property("magmom", orig_magmoms) # add magmom to structure
    struct = struct.get_sorted_structure() # sort structure with the magmoms
    rearranged_magmoms = struct.site_properties['magmom'] # get the rearranged magmoms
    numeric_strings = [str(value) for value in rearranged_magmoms] # convert values to strings
    result_string = ' '.join(numeric_strings) # join the strings
    result_string = "MAGMOM = " + result_string # add the MAGMOM = part

    # Write the result_string to the INCAR file
    with open(incar_file, 'r') as file:
        lines = file.readlines()
    with open(incar_file, 'w') as file:
        for line in lines:
            if line.startswith('MAGMOM ='):
                file.write(result_string + '\n')
            else:
                file.write(line)
    
    # Write the rearranged structure to the POSCAR file
    poscar = Poscar(struct)
    poscar.write_file(poscar_file)
    return None

"""
This function loops through all the config directories and scales
the POSCAR files to the specified volume per atom. It does not return
anything, but it changes the POSCAR files in place.
volums_per_atom (float or int) – The volume per atom in Å^3/atom
"""
def scale_poscars(vol_per_atom, configurations_directory='configurations'):
    config_dirs = [dir for dir in os.listdir(configurations_directory) if os.path.isdir(os.path.join(configurations_directory, dir))]
    for config_dir in config_dirs:
        poscar_file = os.path.join(configurations_directory, config_dir, 'POSCAR')
        struct = Structure.from_file(poscar_file)
        number_of_atoms = struct.num_sites
        volume = vol_per_atom * number_of_atoms
        struct.scale_lattice(volume)
        poscar = Poscar(struct)
        poscar.write_file(poscar_file)
    return None

"""
Changes lreal to false in the incar if it has less than or equal to max_atoms.
This is useful for small systems where the real space projection is not
recommended.
"""
def lreal_to_false(configurations_directory='configurations', max_atoms=10):
    config_dirs = [dir for dir in os.listdir(configurations_directory) if os.path.isdir(os.path.join(configurations_directory, dir))]
    for config_dir in config_dirs:
        poscar_file = os.path.join(configurations_directory, config_dir, 'POSCAR')
        struct = Structure.from_file(poscar_file)
        if struct.num_sites <= max_atoms:
            incar_file = os.path.join(configurations_directory, config_dir, 'INCAR')
            with open(incar_file, 'r') as file:
                lines = file.readlines()
            with open(incar_file, 'w') as file:
                lreal_found = False
                for line in lines:
                    if line.startswith('LREAL ='):
                        file.write('LREAL = .FALSE.\n')
                        lreal_found = True
                    else:
                        file.write(line)
                if not lreal_found:
                    file.write('LREAL = .FALSE.\n')
    return None

def change_incar_tag(tag, value, configurations_directory='configurations'):
    config_dirs = [dir for dir in os.listdir(configurations_directory) if os.path.isdir(os.path.join(configurations_directory, dir))]
    for config_dir in config_dirs:
        incar_file = os.path.join(configurations_directory, config_dir, 'INCAR')
        with open(incar_file, 'r') as file:
            lines = file.readlines()
        with open(incar_file, 'w') as file:
            tag_found = False
            for line in lines:
                if line.startswith(tag):
                    file.write(f'{tag} = {value}\n')
                    tag_found = True
                else:
                    file.write(line)
            if not tag_found:
                file.write(f'{tag} = {value}\n')
    return None


"""
df here is a dataframe that contains a 'config' column of the configs you want to
set up the ev calculation for.
"""
def set_up_ev_from_fixed_volume_calculations(df, path_to_fixed_volume_configurations, dest_configurations_path):
    try:
        os.makedirs(dest_configurations_path, exist_ok=True)
    except FileExistsError:
        print(f"Destination directory '{dest_configurations_path}' already exists.")
        return None

    for config in df['config']:
        subdirectories = [d for d in os.listdir(os.path.join(path_to_fixed_volume_configurations, f'config_{config}')) if os.path.isdir(os.path.join(path_to_fixed_volume_configurations, f'config_{config}', d))]        
        if len(subdirectories) != 1:
            print(f"ERROR: There should be exactly one subdirectory (e.g., 'vol_21') for each config '{config}' in the dataframe.\nBut there are {len(subdirectories)} subdirectories.")
            return None
        
        subdirectory = subdirectories[0]
        fixed_vol_contcar = os.path.join(path_to_fixed_volume_configurations, f'config_{config}', subdirectory, "CONTCAR")
        fixed_vol_potcar = os.path.join(path_to_fixed_volume_configurations, f'config_{config}', subdirectory, "POTCAR")
        fixed_vol_incar = os.path.join(path_to_fixed_volume_configurations, f'config_{config}', subdirectory, "INCAR")
        
        dest_poscar = os.path.join(dest_configurations_path, f'config_{config}', "POSCAR")
        dest_potcar = os.path.join(dest_configurations_path, f'config_{config}', "POTCAR")
        dest_incar = os.path.join(dest_configurations_path, f'config_{config}', "INCAR")
        
        try:
            os.makedirs(os.path.join(dest_configurations_path, f'config_{config}'), exist_ok=False)
            shutil.copy(fixed_vol_contcar, dest_poscar)
            shutil.copy(fixed_vol_potcar, dest_potcar)
            shutil.copy(fixed_vol_incar, dest_incar)
        except Exception as e:
            print(f"Error processing configuration '{config}': {str(e)}")    
    return None

def write_structure_to_lat_in(
    structure,
    filename="lat.in",
    replace_atoms:dict = {}):

    # Open the file for writing
    with open(filename, 'w') as file:
        # Write an euclidean coordinate system
        file.write('1 1 1 90 90 90\n')
        # Write the lattice vectors
        for vec in structure.lattice.matrix:
            file.write(f"{vec[0]} {vec[1]} {vec[2]}\n")
        
        # Write the atomic positions and types
        for site in structure.sites:
            specie = site.specie.symbol  # Get the species symbol
            coords = site.frac_coords  # Get fractional coordinates
            if specie in replace_atoms:
                specie = replace_atoms[specie]
            file.write(f"{coords[0]} {coords[1]} {coords[2]} {specie}\n")

def run_newgenstrYW(
    args:list,
    path:str,
    poscar_name:str='POSCAR',
    replace_atoms:dict = {}):
    
    os.mkdir(os.path.join(path, 'atat_stuff'))
    poscar_file = os.path.join(path, poscar_name)
    struct = Structure.from_file(poscar_file)
    latin = os.path.join(path, 'atat_stuff', 'lat.in')
    write_structure_to_lat_in(
        struct,
        filename=latin,
        replace_atoms=replace_atoms
    )
    cwd = os.getcwd()
    os.chdir(os.path.join(path, 'atat_stuff'))
    with open('YWoutput', 'w') as outfile:
        subprocess.run(args, stdout=outfile)
    os.chdir(cwd)
    
def generate_magnetic_configs(
    path,
    incar_name,
    poscar_name,
    potcar_name,
    magmoms,
    dummy_species_pairs,
    replace_atoms,
    submit_script,
    newgenstrYW_args= [
        "newgenstrYW",
        "-sig",
        "16",
        "-l",
        "lat.in"
    ],
):
    """_summary_

    Args:
        path (_type_): _description_
        incar (_type_): _description_
        potcar (_type_): _description_
        magmoms (_type_): _description_
        dummy_species_pairs (_type_): _description_
        submit_script (_type_): _description_
        scale_volume (_type_): _description_

    Raises:
        FileExistsError: _description_
    """
    incar = os.path.join(path, incar_name)
    poscar = os.path.join(path, poscar_name)
    potcar = os.path.join(path, potcar_name)
    
    run_newgenstrYW(
        newgenstrYW_args,
        path,
        poscar_name=poscar_name,
        replace_atoms=replace_atoms
    )
    
    yw_output = os.path.join(path, 'atat_stuff', 'YWoutput')
    strs_path = os.path.join(path, 'atat_stuff', 'strs')
    
    strs_dir = os.path.join(path, 'atat_stuff', 'strs')
    parse(yw_output, strs_dir)
    atom_count_df = count_atoms(strs_dir)
    for up_down_pair in dummy_species_pairs:
        remove_spin_up_less_than_down(
            strs_path,
            atom_count_df,
            up_down_pair[0],
            up_down_pair[1]
        )
    configurations_dir = os.path.join(path, 'configurations')
    if not os.path.exists(configurations_dir):
        os.makedirs(configurations_dir)
    else:
        raise FileExistsError(
            f"Directory {configurations_dir} already exists. "
            "Please remove it and try again."
        )
    convert_strs_to_poscars(strs_dir, configurations_dir)
    make_incars(magmoms, incar, configurations_dir, strs_dir)
    for dir in os.listdir(configurations_dir):
        shutil.copy(potcar, os.path.join(configurations_dir, dir, 'POTCAR'))
        rearrange_sites_and_magmoms(os.path.join(configurations_dir, dir))
    scale_poscars(10, configurations_dir)
    make_kpoints(600, configurations_dir)
    create_submit_scripts(
        configurations_directory=configurations_dir,
        submit_script=submit_script
    )

"""
End of section
"""