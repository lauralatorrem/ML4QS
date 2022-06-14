import pandas as pd
import numpy as np
import re
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plot
import matplotlib.dates as md
from util.util import get_chapter

import matplotlib.colors as cl
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram
import itertools
from scipy.optimize import curve_fit
import re
import math
import sys
from pathlib import Path
import dateutil
import matplotlib as mpl
mpl.use('tkagg')

df = pd.read_csv('./intermediate_datafiles/phyphox2_result.csv')

df["labelOnTable"] = 0
df["labelStanding"] = 0
df["labelWalking"] = 0
df["labelCooking"] = 0
df["labelDancing"] = 0
df["labelRunning"] = 0
df["labelSitting"] = 0

df.loc[0:236, ["labelOnTable"]] = 1
df.loc[237:531, ["labelStanding"]] = 1
df.loc[532:753, ["labelWalking"]] = 1
df.loc[754:1195, ["labelCooking"]] = 1
df.loc[1196:1690, ["labelDancing"]] = 1
df.loc[1691:1966, ["labelRunning"]] = 1
df.loc[1967:2479, ["labelSitting"]] = 1

df.to_csv('./intermediate_datafiles/phyphox2_labels.csv')


