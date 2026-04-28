import pandas as pd
import glob
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units

c='20190815'
c='20100912'
c='20060111'
c='20070125'
c='20080313'
c='20061007'
f=glob.glob(f'../data/*{c}*.txt')[0]
df = pd.read_csv(f, delim_whitespace=True, skiprows=0)

p = df["P[hPa]"].values * units.hPa
T = df["T[K]"].values * units.kelvin
qv = df["Qv[g/kg]"].values * units('g/kg')
u = df["U[m/s]"].values * units('m/s')
v = df["V[m/s]"].values * units('m/s')

T_C = T.to('degC')

from metpy.calc import mixing_ratio_from_specific_humidity, dewpoint_from_specific_humidity

#convert Td
dewpoint = dewpoint_from_specific_humidity(p, T, qv)

#Draw SkewT
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=45)

#draw T and Td
skew.plot(p, T_C, 'goldenrod', linewidth=2, label="Temperature")
skew.plot(p, dewpoint, 'm', linewidth=2, label="Dew Point")

#draw Wind
skew.plot_barbs(p, u, v, xloc=1.07)

skew.plot_dry_adiabats(colors='grey')
skew.plot_moist_adiabats(colors='grey')
skew.plot_mixing_lines(colors='grey')

plt.ylim(1050, 100)
plt.xlim(-40, 40)
plt.legend()
plt.gca().set_xlabel('Temperature [$^\circ$C]')
plt.gca().set_ylabel('Pressure [hPa]')
plt.title(c, fontsize=14)
plt.savefig(f'../figures/snd.{c}.png',dpi=200)
plt.show()

