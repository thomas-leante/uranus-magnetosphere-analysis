# %% [markdown]
# La creazione di questo breve script è finalizzata all'ottenimento e presentazione di una funzione in grado di calcolare la latitudine magnetica (MLAT) della sonda Voyager 2 a partire dalle sue coordinate cartesiane rispetto la terna di riferimento IAU URANUS.
# 
# Il seguente lavoro si dividerà in due parti:
# - Una prima parte in cui, tramite apposite librerie, verranno estratti i dati (Speasy) e calcolati gli input (SpiceyPy) necessari all'impiego della funzione;
# - Una seconda parte in cui verrà presentata la funzione desiderata con le sue specifiche; 
# 
# Al termine delle due parti verrà poi aggiunta, in una sezione apposita, un'ulteriore funzione di conversione di coordinate. L'ulteriore proposta servirà come strumento di confronto utile alla validazione della soluzione individuata e permetterà di osservare alcune particolarità sulle differenza di ideazione e scrittura.

# %% [markdown]
# PRIMA PARTE:

# %%
# Library Setup
import speasy as spz
import pandas as pd
import numpy as np
import spiceypy as spice

# %%
def veryspeasy(folder, startdata, endata):
    """
    Simplified access to Speasy data retrieval and concatenation.
    
    This function provides a streamlined interface to fetch data from Speasy
    and combine multiple variables into a single pandas DataFrame.
    
    Args:
        folder (str): Speasy data folder or product identifier
        startdata (array): Start date as (day, month, year) array
        endata (array): End date as (day, month, year) array
        
    Returns:
        pd.DataFrame: Combined dataframe containing all variables from the 
                     specified folder concatenated along columns
    """
    
    # Convert date array to ISO format strings required by Speasy
    # Format: "YYYY-MM-DD" from (day, month, year) array
    start = f"{startdata[2]}-{startdata[1]}-{startdata[0]}"  # Year-Month-Day
    end = f"{endata[2]}-{endata[1]}-{endata[0]}"  # Year-Month-Day
    
    # Retrieve data from Speasy using the specified folder and date range
    data_of_folder = spz.get_data(folder, start, end)
    
    # Initialize list to store individual variable dataframes
    df_list = []
    
    # Process each variable in the retrieved dataset
    for v in data_of_folder.variables:
        # Extract data for the current variable
        v_data = data_of_folder.variables[v]
        
        # Convert variable data to pandas DataFrame
        df = v_data.to_dataframe()
        
        # Add the variable's dataframe to our collection
        df_list.append(df)
    
    # Concatenate all variable dataframes along columns (axis=1)
    # This creates a single dataframe with each variable as a separate column
    DF = pd.concat(df_list, axis=1)
    
    return DF

# %%
# -----------------------
# Access to AMDA datatree:
# -----------------------

# 1. Remove other data providers exept AMDA in order to retrieve data faster:
spz.config.core.disabled_providers.set('sscweb,cdaweb,csa') 
# 2. Access to datatree:
amda_tree = spz.inventories.tree.amda

# ---------------------------------
# Access to PLS data from Voyager 2:
# ---------------------------------
electron_moments=amda_tree.Parameters.Voyager.Voyager_2.PLS.flyby_uranus.vo2_pls_uraelefit
df_electron=veryspeasy(electron_moments,[24,1,1986],[25,1,1986])

# %%
def get_position(dataframe):
    """
    Calculate Voyager 2 spacecraft positions relative to Uranus and merge with input dataframe.
    
    This function uses NASA's SPICE toolkit to compute spacecraft positions in both Cartesian
    and spherical coordinates, normalizes them by Uranus' radius, and joins the results
    with the input dataframe.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe with datetime index containing original data
        
    Returns:
        pd.DataFrame: Original dataframe augmented with position columns:
            - pos_x, pos_y, pos_z: Normalized Cartesian coordinates in IAU URANUS
            - r: Radial distance (normalized)
            - long: Longitude in degrees
            - lat: Latitude in degrees
    """
    
    # Load SPICE kernel file for planetary ephemeris data
    spice.furnsh("vo2MetaK.txt")
    
    # Initialize empty lists for Cartesian coordinates (normalized by Uranus radius)
    x_lista_coordinate = []
    y_lista_coordinate = []
    z_lista_coordinate = []
    
    # Initialize empty lists for spherical coordinates
    r_lista = []
    lon_lista = []
    lat_lista = []

    # Extract timestamps from dataframe index for position calculations
    timespan = list(dataframe.index)
    
    # Calculate positions for each timestamp
    for t in timespan:
        # Convert pandas timestamp to Python datetime for SPICE compatibility
        tc = t.to_pydatetime()
        
        # Convert datetime to ephemeris time (ET) used by SPICE
        et = spice.datetime2et(tc)
        
        # Calculate Cartesian position of Voyager 2 relative to Uranus
        # Using IAU_URANUS frame without light-time correction
        pos, _ = spice.spkpos("VOYAGER 2", et, "IAU_URANUS", "NONE", "URANUS BARYCENTER")
        
        # Normalize coordinates by Uranus radius (25559 km) for unitless values
        pos_ru = pos / 25559.0
        
        # Store normalized Cartesian coordinates
        x_lista_coordinate.append(pos_ru[0])
        y_lista_coordinate.append(pos_ru[1])
        z_lista_coordinate.append(pos_ru[2])
        
        # Convert Cartesian coordinates to spherical coordinates
        r, lon, lat = spice.reclat(pos_ru)
        
        # Store spherical coordinates, converting radians to degrees for longitude/latitude
        r_lista.append(r)
        lon_lista.append(np.degrees(lon))
        lat_lista.append(np.degrees(lat))
    
    # Create dataframe with calculated position data
    data = {
        "delta t": timespan,
        "x": x_lista_coordinate,
        "y": y_lista_coordinate, 
        "z": z_lista_coordinate,
        "r": r_lista,
        "long": lon_lista,
        "lat": lat_lista
    }
    df = pd.DataFrame(data)
    
    # Set datetime index to match original dataframe structure
    df["delta t"] = pd.to_datetime(df["delta t"])
    df = df.set_index("delta t")
    
    # Rename Cartesian coordinate columns for clarity
    df = df.rename(columns={
        "x": "pos_x",
        "y": "pos_y", 
        "z": "pos_z"
    })
    
    # Merge position data with original dataframe
    dataframe = dataframe.join(df)
    
    # Clear SPICE kernel to free memory
    spice.kclear()
    
    return dataframe

# %%
# Apply 'get_position()' function to 'df_electron' dataframe in order to evaluate Voyager 2 position during PLS activity:
df_electron=get_position(df_electron) 
# Extract needed inputs for MLAT estimate:
x_iau,y_iau,z_iau=df_electron['pos_x'],df_electron['pos_y'],df_electron['pos_z']

# %% [markdown]
# SECONDA PARTE:

# %% [markdown]
# Ottenuti tutti gli input necessari, viene presentata adesso la funzione 'cartesian_to_magnetic_uranus'.
# Le specifiche inerenti il suo funzionamento sono riportatate nell'apposita sezione commentata, tuttavia è doveroso qui fare alcune premesse.
# La sua ideazione e scrittura sono fondate interamente sul modello OTD (Offset Tilted Dipole), sviluppato dal team di Ness et al. nel 1986 sulla base dei dati della sonda Voyager 2. Questo modello rappresentò all'epoca la prima e più fondamentale approssimazione del campo magnetico uraniano mai ottenuta.
# 
# Prima del flyby di Voyager 2 nel 1986, la comunità scientifica ignorava completamente l'esistenza e le caratteristiche del campo magnetico di Urano. La scoperta di un campo magnetico intrinseco fu di per sé rivoluzionaria, ma ciò che rese questa rivelazione ancora più straordinaria furono le sue proprietà uniche:
# - Inclinazione estrema: 60° tra asse magnetico e asse di rotazione
# - Dipolo decentrato: Spostato di 0.3 Rᵤ dal centro del pianeta
# - Asimmetria emisferica: Campo superficiale variabile tra 0.1 e 1.1 Gauss
# 
# In un'epoca in cui si conoscevano solo campi magnetici quasi-assialsimmetrici (Terra, Giove, Saturno), il modello OTD costituì dunque un progresso concettuale fondamentale, fornendo per la prima volta uno strumento matematico per descrivere questa anomalia magnetica.
# 
# Il codice implementa fedelmente le approssimazioni originali del modello:
# - Approssimazione di dipolo puro: Il campo magnetico completo viene ridotto alle sole componenti dipolari
# - Traslazione rigida: Il centro del dipolo viene semplicemente spostato rispetto al centro planetario
# - Geometria vettoriale semplice: Le conversioni di coordinate si basano su trasformazioni geometriche elementari
# 
# È importante sottolineare che oggi sappiamo che il modello OTD rappresenta una semplificazione significativa della reale complessità del campo magnetico uraniano. Successivi approfondimenti (Connerney et al., 1987) hanno rivelato:
# - Contributi multipolari significativi: I termini di quadrupolo e ottupolo sono comparabili al dipolo vicino alla superficie
# - Struttura complessa: Il campo reale presenta anomalie non catturate dal semplice OTD
# - Modelli più sofisticati: Il successivo modello Q3 ha fornito una rappresentazione più accurata
# 
# Tuttavia, questa evoluzione delle conoscenze verrà qui trascurata frenando le proprie aspirazioni ad un calcolo più facile e intuitivo.

# %%
def acos(x):
    """
    Safe arccosine function with numerical stability.
    
    Wraps numpy arccosine with clipping to prevent numerical errors
    from values slightly outside [-1, 1] due to floating point precision.
    
    Args:
        x (float or array-like): Input value(s) for arccosine calculation
        
    Returns:
        float or array: Arccosine of input, with values clipped to valid range
    """
    return np.arccos(np.clip(x, -1.0, 1.0))


def cartesian_to_magnetic_uranus(x_iau, y_iau, z_iau):
    """
    Convert IAU Uranus cartesian coordinates to magnetic coordinates using OTD model.
    
    This function implements the Offset Tilted Dipole (OTD) magnetic field model
    for Uranus based on Voyager 2 measurements from Ness et al. (1986). It calculates
    magnetic latitude and colatitude for positions in the IAU Uranus coordinate system.
    
    -----------
    Parameters:
    -----------
    x_iau, y_iau, z_iau : float or array-like
        Cartesian coordinates in IAU Uranus frame (planet-centered)
        Units are in Uranus radii (normalized)
    
    --------
    Returns:
    --------
    dict with arrays of:
        magnetic_latitude : float (degrees, -90 to +90)
            Magnetic latitude relative to magnetic equator
        magnetic_colatitude : float (degrees, 0 to 180)
            Angle from magnetic north pole (0° at pole, 180° at opposite pole)
        distance_from_dipole : float (Uranus radii)
            Distance from the offset dipole center
    """
    
    # OTD model parameters from Ness et al. (1986) Voyager 2 analysis
    # Dipole tilt angle relative to rotation axis
    DIPOLE_TILT = np.radians(60.0)
    
    # Magnetic pole coordinates in IAU Uranus frame
    # Latitude and longitude of the magnetic north pole
    POLE_LAT = np.radians(15.2)    # Planetographic latitude in radians
    POLE_LON = np.radians(360 - 47.7)  # Planetographic longitude in radians (312.3°)
    
    # Calculate magnetic pole unit vector in IAU coordinates
    # Converts spherical pole coordinates to Cartesian unit vector
    mag_pole_x = np.cos(POLE_LAT) * np.cos(POLE_LON)
    mag_pole_y = np.cos(POLE_LAT) * np.sin(POLE_LON) 
    mag_pole_z = np.sin(POLE_LAT)
    
    # Dipole offset parameters (in planetary radii)
    # The magnetic dipole is offset from the planet center
    DIPOLE_OFFSET_X = -0.02   # Offset in x-direction (Uranus radii)
    DIPOLE_OFFSET_Y = 0.02    # Offset in y-direction (Uranus radii)
    DIPOLE_OFFSET_Z = -0.31   # Offset in z-direction (Uranus radii)
    
    # Apply dipole offset: shift coordinates to dipole-centered system
    # Transform from planet-centered to dipole-centered coordinates
    x_offset = x_iau - DIPOLE_OFFSET_X
    y_offset = y_iau - DIPOLE_OFFSET_Y
    z_offset = z_iau - DIPOLE_OFFSET_Z
    
    # Calculate position vector magnitude (distance from dipole center)
    # Uses np.sqrt instead of math.sqrt to handle arrays efficiently
    r_mag = np.sqrt(x_offset**2 + y_offset**2 + z_offset**2)
    
    # Normalize position vector to unit length
    # Initialize arrays with zeros to handle edge cases
    x_norm = np.zeros_like(x_offset)
    y_norm = np.zeros_like(y_offset)
    z_norm = np.zeros_like(z_offset)
    
    # Apply normalization only where r_mag > 0 to avoid division by zero
    # Creates boolean mask for valid positions
    mask = r_mag > 0
    x_norm[mask] = x_offset[mask] / r_mag[mask]
    y_norm[mask] = y_offset[mask] / r_mag[mask]
    z_norm[mask] = z_offset[mask] / r_mag[mask]
    
    # Calculate magnetic colatitude using dot product with pole direction
    # cos(colatitude) = (position_vector · magnetic_pole_vector)
    cos_colat = (x_norm * mag_pole_x + 
                 y_norm * mag_pole_y + 
                 z_norm * mag_pole_z)
    
    # Ensure numerical stability by clipping to valid cosine range
    # Prevents NaN values from floating point precision errors
    cos_colat = np.clip(cos_colat, -1.0, 1.0)
    
    # Calculate magnetic colatitude (angle from magnetic pole)
    # colatitude = 0° at magnetic pole, 180° at opposite pole
    magnetic_colat_rad = acos(cos_colat)
    
    # Calculate magnetic latitude (90° - colatitude)
    # latitude = -90° at south pole, +90° at north pole
    magnetic_lat_rad = np.pi/2 - magnetic_colat_rad
    magnetic_latitude = np.degrees(magnetic_lat_rad)
    
    return {
        'magnetic_latitude': magnetic_latitude,
        'magnetic_colatitude': np.degrees(magnetic_colat_rad),
        'distance_from_dipole': r_mag
    }

# %%
# Calculate MLAT according to OTD magnetic representation:
magnetic_voy2_properties=cartesian_to_magnetic_uranus(x_iau,y_iau,z_iau)
# Convert datatype to dataframe for a better visualization:
magnetic_voy2_properties=pd.DataFrame.from_dict(magnetic_voy2_properties)

# %% [markdown]
# CONFRONTO FINALE:

# %% [markdown]
# Come accennato all'inizio viene adesso presentata un'ulteriore funzione di cambio coordinate, la cui ideazione differisce però dalla precedente. Di seguito vengono riportate le principali differenze individuate e le motivazioni per cui non è stato scartato a confronto di 'cartesian_to_magnetic_uranus':
# 
# 1. Input: Coordinate Cartesiane IAU in km;
# 
# 2. Parametri del Polo Magnetico: 
#     * cartesian_to_magnetic_uranus:
#         - Polo positivo: lat = +15.2°, lon = 47.7°W (convertito in longitudine Est).
#         - Questi sono i valori riportati da Ness et al. (1986) per il modello OTD.
#     * umod_mlat:
#         - Usa anch’esso lat = 15.2°, lon = 47.7°, ma attenzione: qui la longitudine è presa direttamente come 47.7° Est, mentre nel modello di Ness è 47.7° Ovest.
#         Quindi già c’è una discrepanza di ~95° nella direzione del polo magnetico.
# 
# 3. Offset del Dipolo:
#     * cartesian_to_magnetic_uranus: 
#         - Offset esplicito $=(-0.02,+0.02,-0.31) R_u$ preso direttamente dal modello OTD;
#     * umod_mlat:
#         - Offset semplificato $=-0.3 R_u\hat{m}$ (dipolo viene traslato lungo il proprio asse di una frazione di raggio)
# 
# 4. Formula: 
#     * cartesian_to_magnetic_uranus:
#         - Calcola prima la colatitudine magnetica (angolo fra $\hat{r}$ e $\hat{m}$) e poi la trasforma in latitudine magnetica.
#         - $\varphi_{m}=90°-\arccos(\hat{r}'\cdot\hat{m}) $
#     * umod_mlat: 
#         - usa direttamente: $\sin(\varphi_{m})=\frac{(r')\cdot\hat{m}}{|r'|}$
#         - Questa è matematicamente equivalente solo se l’offset è lungo l’asse magnetico. Ma siccome l’offset nei due codici è diverso, anche la formula porta a valori diversi.

# %%
def umod_mlat(x, y, z):

    """
    Compute Uranian magnetic latitude (MLAT) for points in the IAU Uranus frame.

    ----------
    Parameters
    ----------

    x, y, z : array_like
        Cartesian coordinates of the point(s) in kilometers. Can be scalars or NumPy arrays
        of the same shape.

    -------
    Returns
    -------

    mlat : ndarray or float
        Magnetic latitude in degrees, same shape as inputs.
    """

    # Uranus model constants
    RU = 25559.0                   # Uranus radius in km
    lat_m = np.radians(15.2)       # North magnetic pole latitude in radians
    lon_m = np.radians(47.7)       # North magnetic pole longitude in radians
    # Unit vector along magnetic axis (m̂)
    mhat = np.array([
        np.cos(lat_m) * np.cos(lon_m),
        np.cos(lat_m) * np.sin(lon_m),
        np.sin(lat_m)
    ])
    # Dipole center offset vector O = −0.3 RU · m̂
    offset = -0.3 * RU * mhat     # shape (3,)
    # Translate input points by offset
    xp = np.asarray(x) - offset[0]
    yp = np.asarray(y) - offset[1]
    zp = np.asarray(z) - offset[2]
    # Dot product r′·m̂ and magnitude |r′|
    dot = xp * mhat[0] + yp * mhat[1] + zp * mhat[2]
    norm = np.sqrt(xp**2 + yp**2 + zp**2)
    # Argument for arcsin, clamped to [−1, +1]
    arg = np.clip(dot / norm, -1.0, 1.0)
    # Magnetic latitude in degrees
    mlat = np.degrees(np.arcsin(arg))
    return mlat

# %%
# Calculate MLAT according to simplified magnetic representation:
Ru=25559.0 # Uranian Radius for data conversion
simply_magnetic_voy2_properties=umod_mlat(x=x_iau*Ru,y=y_iau*Ru,z=z_iau*Ru)

# Import all data into a dataframe with same index as df_electron for a better visualization and comparison:
data={"delta_t":list(df_electron.index),"simply_MLAT":simply_magnetic_voy2_properties}
simply_magnetic_voy2_properties=pd.DataFrame(data)
# Set index correctly in order to match with df_electron:
simply_magnetic_voy2_properties["delta_t"] = pd.to_datetime(simply_magnetic_voy2_properties["delta_t"])
simply_magnetic_voy2_properties = simply_magnetic_voy2_properties.set_index("delta_t")
# Delete useless items:
del data

# %%
# Results Comparison:
print(f'First 3 items of magnetic_voy2_properties dataframe:\n', magnetic_voy2_properties['magnetic_latitude'].iloc[:3])
print('\n')
print('------------------------------------------------------------------------------------------')
print(f'First 3 items of simply_magnetic_voy2_properties dataframe:\n',simply_magnetic_voy2_properties['simply_MLAT'].iloc[:3])


