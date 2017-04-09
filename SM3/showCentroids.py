from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

params = {'figure.subplot.left': 0.0,
          'figure.figsize': (10, 10),
          'figure.subplot.right': 1.0,
          'figure.subplot.bottom': 0.0,
          'figure.subplot.top': 1.0}
plt.rcParams.update(params)

fig = plt.figure()
ax = fig.gca(projection='3d')

A10 = np.array([[10.70014771, 11.816839, 5.93426883], [177.93280977, 189.14048866, 202.16972077],
                [146.00965575, 183.50209908, 216.48320739], [4.53282828, 10.29461279, 1.38468013],
                [126.83273381, 138.68480216, 148.51303957], [126.42133423, 130.43872262, 128.54138104],
                [35.34364995, 46.16414088, 13.98932764], [121.0697928, 157.63249727, 193.90076336],
                [164.39690223, 164.1755405, 156.81736044], [137.74125874, 138.44631839, 132.66104484],
                [110.53516613, 145.14757969, 182.23916343], [191.16788732, 193.98338028, 197.05605634],
                [116.96948448, 151.90793403, 187.92001039], [117.90320553, 156.00942803, 192.71527341],
                [3.30141844, 2.72948328, 1.49088146], [29.28755538, 30.81876762, 29.57511075],
                [127.40595552, 165.46814926, 200.37881643], [146.95851918, 136.01605709, 115.95138269],
                [119.32852318, 122.13601474, 119.20448265], [237.76703992, 234.0744888, 229.82302824],
                [62.42119741, 62.32783172, 51.98349515], [20.05408728, 30.47971727, 6.05961893],
                [112.70824868, 115.4002438, 112.41568468], [81.5619322, 96.7586971, 109.8539774],
                [45.96238852, 54.76179398, 22.55538322], [155.57922669, 162.78999242, 167.13722517],
                [14.27757353, 23.19485294, 4.74126838], [97.57275772, 100.08931894, 94.4369185],
                [135.53480475, 124.90933786, 104.8057725], [125.87517053, 159.96111869, 194.],
                [147.28688293, 151.92383639, 151.25416079], [126.15748031, 162.03412073, 196.68678915],
                [28.03641661, 39.44403982, 9.11944647], [130.6362379, 158.69294606, 187.92392808],
                [41.08285303, 36.98414986, 26.56664265], [150.54974835, 81.21718931, 62.74448316],
                [122.17583989, 160.05218013, 196.58184417], [123.67362146, 157.77198212, 192.35842027],
                [51.03502606, 77.75312083, 103.81917343], [114.94983819, 124.01294498, 127.68608414],
                [123.05617978, 155.95764909, 189.74200519], [109.21274991, 117.31233497, 119.30554508],
                [157.62067395, 176.87249545, 198.98178506], [175.82402325, 177.75298676, 176.62867291],
                [21.82839721, 17.02090592, 9.1315331], [135.89919485, 171.75458937, 203.51626409],
                [138.73931178, 166.0354536, 194.28519291], [144.79417353, 146.00443319, 140.5440152],
                [26.16691395, 26.29747774, 15.39651335], [249.67197021, 248.49627624, 247.5267434],
                [102.48342682, 109.17746048, 110.04742478], [210.25844076, 209.56824228, 208.34131369],
                [86.55682878, 84.9491006, 62.70872751], [163.90710383, 152.64480874, 134.42622951],
                [62.37373737, 74.78442966, 26.79958118], [132.59677419, 138.94310036, 139.02105735],
                [120.34511093, 154.42152835, 189.36565325], [14.14183835, 16.28526149, 16.07527734],
                [118.93173002, 114.26842514, 97.30488751], [188.81781609, 187.10689655, 179.16091954],
                [116.03609987, 103.12820513, 79.96963563], [36.65621698, 44.2226136, 51.06586826],
                [84.18081494, 109.69439728, 131.98344652], [9.47633588, 17.75648855, 1.5778626]])

A30 = np.array([[104.9372608, 94.91074358, 75.91375068], [131.46120069, 168.32615894, 201.47077455],
                [132.22629137, 136.15502629, 134.09712341], [166.95008726, 84.81640489, 70.03246073],
                [139.68826123, 146.8348383, 149.23687526], [123.10363005, 156.08708128, 189.82777259],
                [95.07719635, 104.09591935, 108.25828133], [3.23443927, 3.77748835, 1.46490266],
                [130.17829619, 122.09747969, 103.80233285], [105.43767507, 104.85381653, 95.91771709],
                [49.88050655, 63.1572681, 16.3515532], [73.32568744, 69.88047534, 52.87563908],
                [60.65776956, 69.94556025, 34.22859408], [63.2725265, 77.13147821, 22.85453475],
                [15.86440678, 25.15293426, 4.84079553], [114.07631507, 119.40228945, 118.98288362],
                [138.85766423, 164.94251825, 192.65723844], [61.53757724, 65.75082627, 64.38654979],
                [107.30679513, 112.48514923, 111.72326862], [114.38019379, 149.10050911, 185.71013303],
                [128.4267658, 129.21157727, 123.40839087], [178.57442748, 179.16977665, 175.24081142],
                [107.83521584, 142.60003095, 180.28345969], [161.3951505, 158.09464883, 147.57625418],
                [39.6998016, 50.89327495, 16.07764931], [64.51074503, 90.50864348, 115.03165887],
                [14.57988705, 14.51029331, 7.08307524], [7.94702824, 14.187017, 2.22411128],
                [120.06161396, 126.83112634, 127.98496651], [82.02248062, 81.29091916, 69.34462901],
                [49.31633448, 75.53931797, 101.20001557], [124.50158869, 160.55161672, 195.77154071],
                [37.08171767, 44.81204935, 51.68787053], [147.0062397, 132.93477749, 109.25665175],
                [111.06089131, 145.62612149, 183.01397264], [92.49660397, 105.79075235, 46.63218391],
                [183.84925105, 190.64703415, 198.56722588], [113.37735849, 143.33619211, 170.54202401],
                [116.24477854, 152.05104429, 188.59488657], [49.2786816, 58.6447285, 26.67050497],
                [75.53242982, 88.36066934, 34.79186834], [25.69874063, 23.94113485, 13.41771615],
                [242.33879017, 239.8059792, 237.02112775], [29.5654703, 29.36138614, 25.48007426],
                [83.72697595, 110.22955326, 133.61975945], [149.2693294, 179.58045903, 208.16027696],
                [146.14699956, 145.79798511, 138.64240035], [119.01151034, 156.66837766, 193.31591227],
                [15.01603725, 16.98706674, 17.03828246], [143.28158529, 80.86446245, 59.93453033],
                [207.69138172, 207.09891835, 205.48551989], [118.31731641, 151.27923844, 185.98957389],
                [23.6634038, 34.31594966, 7.82322942], [31.60348189, 41.88384401, 12.46908078],
                [56.53974122, 52.26274096, 40.78558226], [109.8902148, 148.29594272, 186.21002387],
                [157.27446923, 161.42407341, 160.25881612], [112.04022191, 64.50485437, 56.76098012],
                [125.46111919, 137.64553052, 147.46802326], [87.15422127, 92.19430052, 93.50792441],
                [36.98529094, 61.96322734, 87.39952412], [39.87094163, 38.16305423, 31.07086968],
                [119.38574953, 153.96377204, 189.31431666], [124.20392369, 111.56677466, 87.55147588]])

B02 = np.array([[34.86363636, 39.23295455, 42.70170455], [5.47530864, 8.51028807, 2.00411523],
                [215.01083032, 213.67509025, 211.77617329], [248.43707094, 246.67505721, 244.62929062],
                [121.78233438, 155.61514196, 190.01577287], [76.59047619, 95.2952381, 113.95238095],
                [58.93939394, 70.02525253, 25.62121212], [233.98032787, 230.32459016, 226.5704918],
                [83.96987952, 110.52409639, 134.3253012], [152.86842105, 186.61403509, 217.03070175],
                [33.84280303, 46.23295455, 11.8655303], [192.65454545, 195.32207792, 197.63506494],
                [109.97021277, 120.29787234, 124.85957447], [126.56324111, 92.93083004, 73.39525692],
                [60.82122905, 78.94972067, 99.27932961], [66.08368201, 92.52301255, 117.25104603],
                [20.43902439, 27.31707317, 13.63414634], [128.7675841, 139.03975535, 146.03975535],
                [50.27717391, 62.70652174, 17.9076087], [82.7755102, 96.08163265, 43.49659864],
                [138.25974026, 174.66666667, 206.45021645], [77.36516854, 78.28089888, 69.29213483],
                [47.78169014, 58.88028169, 22.96478873], [116.8776509, 151.77650897, 187.83686786],
                [23.78448276, 34.38793103, 12.90517241], [27.57936508, 23.07936508, 14.77777778],
                [149.14230769, 135.46923077, 111.83461538], [92.46453089, 98.5583524, 100.01144165],
                [15.97833935, 16.66064982, 12.57761733], [32.04867257, 30.55752212, 26.0619469],
                [130.78333333, 157.86666667, 186.23333333], [56.62790698, 71.89922481, 16.00775194],
                [144.76036866, 144.69815668, 138.75806452], [127.24305556, 126.29861111, 118.59027778],
                [115.98507463, 116.2238806, 109.78358209], [127.72496025, 132.51669316, 131.7281399],
                [49.19362745, 50.26715686, 34.8995098], [109.18779343, 143.7370892, 180.99765258],
                [16.11398964, 24.30051813, 4.80829016], [189.77777778, 186.91358025, 175.7654321],
                [30.91202346, 35.59237537, 13.82697947], [142.16721311, 148.99672131, 149.8852459],
                [125.395, 128.88, 124.63], [135.44020356, 166.92366412, 197.12977099],
                [96.38235294, 95.51176471, 68.23529412], [65.88967136, 63.84976526, 54.40610329],
                [118.79746835, 122.75316456, 121.13924051], [159.0070922, 159.61229314, 153.53427896],
                [20.95131086, 33.46067416, 5.00374532], [41.84960422, 51.32717678, 18.0237467],
                [104.04601227, 110.09509202, 108.77607362], [160.23161765, 178.10661765, 198.75735294],
                [41.73469388, 69.70408163, 97.0994898], [142.21978022, 113.3003663, 90.52747253],
                [101.99009901, 73.15841584, 59.79207921], [126.24064171, 162.7459893, 197.64438503],
                [120.78571429, 158.83766234, 195.51623377], [124.4015748, 158.79527559, 193.37007874],
                [63.62046205, 73.98349835, 34.79867987], [78.08928571, 92.5625, 28.1875, ],
                [54.22857143, 83.95714286, 112.57142857], [97.37272727, 108.03636364, 113.90909091],
                [173.41233141, 174.80924855, 171.82273603], [109.67266187, 115.49640288, 115.54316547]])

B03 = np.array([[35.05154639, 40.11782032, 44.61266568], [5.46576763, 8.32780083, 2.0093361],
                [214.17525773, 212.8910162, 211.28865979], [248.54209919, 246.87081892, 244.88004614],
                [121.46043165, 155.32302158, 189.85035971], [76.00471698, 96.20283019, 115.48584906],
                [58.73394495, 69.81422018, 26.0733945], [234.06666667, 230.35726496, 226.48717949],
                [84.31736527, 111.2245509, 134.96706587], [151.93564356, 186.48762376, 217.51485149],
                [33.82395833, 46.23125, 11.65, ], [191.81077038, 195.09124907, 198.30665669],
                [110.13691932, 120.70415648, 125.39608802], [131.07692308, 91.10677382, 71.58668197],
                [61.13661202, 79.31147541, 99.75956284], [66.00447427, 92.86129754, 117.68456376],
                [20.45679012, 26.72222222, 13.83333333], [128.79904306, 138.97129187, 145.92185008],
                [49.91504854, 62.7961165, 16.98786408], [84.26182965, 97.15772871, 44.21451104],
                [139.14340344, 175.1376673, 206.59082218], [77.33575581, 78.1119186, 69.84156977],
                [48.20923913, 58.80434783, 24., ], [116.75287092, 151.73771245, 187.83004134],
                [24.24666667, 34.78, 12.19333333], [27.38545455, 22.92727273, 14.81818182],
                [148.16697936, 134.67729831, 111.15009381], [92.23860911, 98.29616307, 98.89928058],
                [15.66356877, 16.22862454, 12.20260223], [33.13080895, 31.68674699, 26.49053356],
                [131.86764706, 158.44607843, 185.83333333], [57.02941176, 72.61344538, 15.98319328],
                [144.22271224, 144.35722161, 138.36714443], [128.41197183, 126.0528169, 116.85915493],
                [116.93131313, 116.34949495, 107.92727273], [128.44623201, 133.0169348, 131.97121084],
                [50.08819539, 49.61058345, 35.53731343], [109.45454545, 143.95510662, 181.17283951],
                [15.29807692, 23.91071429, 4.39697802], [188.01277955, 186.22364217, 178.68690096],
                [30.95304348, 36.14956522, 14.07652174], [142.84843493, 149.46293245, 150.57990115],
                [125.17892644, 128.68389662, 125.4612326], [136.01792574, 167.21382843, 197.20870679],
                [98.0369515, 96.31177829, 70.37182448], [65.22711058, 63.91557669, 54.99643282],
                [118.53138686, 122.57810219, 121.22919708], [159.23033067, 159.90649943, 153.90535918],
                [21.5443787, 33.71400394, 4.9270217], [41.52738854, 51.14140127, 17.9388535],
                [104.64153846, 109.66923077, 107.52461538], [161.53396226, 178.85283019, 198.99245283],
                [41.35492578, 69.08367072, 96.27530364], [140.53319058, 114.20342612, 90.7987152],
                [103.5258216, 70.93896714, 57.84507042], [126.2464986, 163.33193277, 198.3487395],
                [120.63992537, 158.63246269, 195.30223881], [124.5125, 158.8125, 193.38541667],
                [64.46859083, 74.77419355, 35.17317487], [77.08695652, 91.41106719, 27.81818182],
                [52.80636605, 83.36339523, 112.22546419], [96.60747664, 107.36760125, 112.96884735],
                [173.14876957, 174.60738255, 171.59955257], [109.72698413, 115.72857143, 115.96190476]])

B04 = np.array([[35.08790072, 40.64115822, 45.65977249], [5.39971449, 8.12205567, 1.96788009],
                [213.70876532, 212.6399623, 211.10273327], [248.6009245, 246.96764253, 244.98613251],
                [121.32434902, 155.18501599, 189.77386935], [75.83333333, 96.69491525, 116.72881356],
                [58.66460587, 69.52550232, 26.42658423], [234.2147806, 230.44457275, 226.54387991],
                [84.36788618, 111.53658537, 135.42073171], [151.45796064, 186.490161, 218.02862254],
                [33.77885331, 46.1816828, 11.4087863], [191.86323453, 195.26677067, 198.71450858],
                [110.14991763, 121.01812191, 125.86490939], [134.92602263, 89.56309835, 70.32985205],
                [61.28070175, 79.67719298, 100.25964912], [65.92455621, 92.72189349, 117.54142012],
                [20.40322581, 26.64516129, 13.84193548], [128.71274298, 139.0075594, 146.05291577],
                [49.88727858, 62.86151369, 16.69404187], [84.58230453, 97.24897119, 44.64197531],
                [139.28148148, 175.1345679, 206.62469136], [77.26778656, 78.02766798, 69.94466403],
                [48.27378965, 58.82136895, 24.61602671], [116.72505614, 151.728906, 187.82932307],
                [24.53125, 35.12890625, 11.73242188], [27.47547974, 22.98933902, 14.96588486],
                [148.25330132, 134.6182473, 111.06242497], [92.33249791, 98.00835422, 98.4160401],
                [15.5394402, 16.00381679, 12.1870229], [33.75475687, 32.07505285, 26.71141649],
                [132.6345515, 158.71428571, 185.73754153], [57.54415954, 73.10541311, 16.06552707],
                [143.90889213, 144.27696793, 138.2951895], [129.0729927, 126.16301703, 115.76155718],
                [117.24641834, 116.21919771, 107.15186246], [128.91418564, 133.36077058, 132.08523059],
                [50.57450628, 49.57809695, 35.9551167], [109.53313697, 144.07731959, 181.15243004],
                [14.94553991, 23.63661972, 4.25915493], [187.15213675, 186.01025641, 180.11282051],
                [30.99511002, 36.4608802, 14.25550122], [143.50588235, 149.84491979, 150.80213904],
                [124.79561201, 128.44110855, 125.93995381], [136.28198198, 167.32522523, 197.15045045],
                [99.16373478, 96.72801083, 71.49255751], [64.76578737, 63.87609912, 55.28776978],
                [118.47912713, 122.5341556, 121.18406072], [159.19209915, 160.02013943, 154.36793184],
                [21.74514877, 33.65847348, 4.80983182], [41.43227425, 51.07023411, 18.06187291],
                [104.96728016, 109.3006135, 106.68813906], [162.21990172, 179.3034398, 199.15847666],
                [41.11303555, 68.76937101, 95.77575205], [138.83830455, 114.95918367, 90.71742543],
                [104.125, 69.75347222, 57.75694444], [126.28787879, 163.64583333, 198.71306818],
                [120.53108808, 158.47279793, 195.13212435], [124.55581669, 158.83901293, 193.35017626],
                [65.07746479, 75.17370892, 35.72769953], [76.44871795, 90.57435897, 28.22051282],
                [52.03535354, 83.0976431, 112.17845118], [96.503663, 107.16666667, 112.44871795],
                [172.81514658, 174.33876221, 171.39983713], [109.82090997, 115.79090029, 116.01161665]])

B05 = np.array([[35.29206349, 41.15793651, 46.3531746], [5.42286947, 8.10787487, 1.97734628],
                [213.48877625, 212.42288197, 210.90224475], [248.59267868, 246.92097618, 244.91284137],
                [121.23320027, 155.10678643, 189.72055888], [75.88439306, 97.21965318, 117.47398844],
                [58.52464403, 69.31763417, 26.79846659], [234.35227273, 230.50699301, 226.51835664],
                [84.62421384, 111.71069182, 135.75157233], [150.91483516, 186.40247253, 218.16895604],
                [33.76806527, 46.21620047, 11.27214452], [191.64900398, 195.19123506, 198.90318725],
                [110.27820513, 121.07307692, 125.96025641], [137.01422475, 88.18776671, 69.17780939],
                [61.39060403, 79.90201342, 100.40939597], [65.94419643, 92.56808036, 117.33035714],
                [20.28216704, 26.49435666, 13.73363431], [128.65525672, 139.00733496, 146.12306438],
                [49.68446026, 62.73309609, 16.43534994], [85.50960118, 98.25553914, 44.84638109],
                [139.34898711, 175.17771639, 206.65469613], [77.32369942, 77.9732659, 69.78540462],
                [48.30821918, 58.76141553, 24.80707763], [116.67523714, 151.6757364, 187.79181228],
                [24.67115903, 35.31536388, 11.48113208], [27.57355126, 23.07280832, 15.18276374],
                [148.20409982, 134.38235294, 110.83600713], [92.30865007, 97.92398427, 98.28636959],
                [15.61247637, 15.85916824, 12.00850662], [33.9825228, 32.43009119, 26.95744681],
                [132.91338583, 159.01312336, 185.97112861], [58.00202429, 73.46761134, 16.31983806],
                [143.73853712, 144.19923581, 138.19268559], [129.44972578, 126.16270567, 115.06764168],
                [117.45588235, 116.08031674, 106.39366516], [129.26360774, 133.55465587, 132.09176788],
                [50.87067138, 49.54699647, 36.23180212], [109.55482456, 144.09429825, 181.14583333],
                [14.77886634, 23.5073478, 4.22043387], [186.53139013, 185.93497758, 180.94394619],
                [31.1091954, 36.6637931, 14.40804598], [143.90343176, 150.08220271, 150.89305666],
                [124.64829822, 128.39303079, 126.07536467], [136.63202247, 167.47191011, 197.1741573],
                [100.32511848, 96.83507109, 72.57535545], [64.69860522, 63.99151001, 55.37537902],
                [118.43621399, 122.45953361, 121.12002743], [159.34900118, 159.98354877, 154.20564042],
                [21.77298851, 33.5210728, 4.82375479], [41.390625, 51.0275, 18.100625],
                [105.24768519, 109.13966049, 106.06635802], [162.70782281, 179.56644675, 199.20640905],
                [41.2554386, 68.66666667, 95.52561404], [137.38647343, 115.17270531, 90.53381643],
                [104.98989899, 69.29040404, 57.74494949], [126.4, 163.84359862, 198.92733564],
                [120.46314741, 158.39541833, 195.06772908], [124.69167354, 158.87798846, 193.28771641],
                [65.32904412, 75.30422794, 35.96875, ], [76.08007449, 90.05586592, 28.5698324],
                [51.55917874, 82.897343, 112.03381643], [96.45736434, 106.97803618, 112.16666667],
                [172.78425096, 174.25608195, 171.11587708], [109.77639752, 115.78536922, 116.0310559]])

B06 = np.array([[35.50678733, 41.57207498, 46.93277311], [5.40182055, 8.03727785, 2.01560468],
                [213.356018, 212.26602925, 210.65410574], [248.59376476, 246.93292395, 244.95229098],
                [121.12858633, 155.05868545, 189.71726656], [75.75565611, 97.47662142, 118.0678733],
                [58.39473684, 69.03947368, 27.12631579], [234.24187975, 230.39460954, 226.40912232],
                [84.8225602, 112.0887199, 136.15969582], [150.62359551, 186.38988764, 218.37865169],
                [33.74009662, 46.25024155, 11.1821256], [191.54426658, 195.14864423, 198.99379288],
                [110.28827878, 121.10982049, 126.08025343], [138.93660287, 87.14952153, 68.34868421],
                [61.34989648, 79.9699793, 100.55072464], [65.81734317, 92.4095941, 117.15774908],
                [20.20538721, 26.41414141, 13.71043771], [128.65974026, 139.08246753, 146.25064935],
                [49.59157509, 62.69139194, 16.25, ], [85.74197384, 98.56837099, 44.93579073],
                [139.48882265, 175.27794337, 206.68479881], [77.33485454, 78.01711352, 69.74614946],
                [48.32417582, 58.70054945, 25.02655678], [116.60502471, 151.61943987, 187.74876442],
                [24.83229814, 35.47308489, 11.21325052], [27.7587822, 23.27400468, 15.37119438],
                [148.15346181, 134.22912206, 110.75874375], [92.24098978, 97.6218397, 98.02635826],
                [15.57055683, 15.74446987, 11.83676583], [34.09295441, 32.65068088, 27.19064535],
                [133.23109244, 159.23529412, 186.08823529], [58.15746753, 73.52272727, 16.58603896],
                [143.639964, 144.12736274, 138.1440144], [129.69185185, 126.1762963, 114.60444444],
                [117.45920304, 116.03036053, 105.9629981], [129.47060945, 133.70248828, 132.15759106],
                [50.98277842, 49.48851894, 36.33352468], [109.58708839, 144.14384749, 181.21837088],
                [14.61315939, 23.39251276, 4.13216109], [186.14739414, 185.85016287, 181.27605863],
                [31.17279124, 36.8663018, 14.62548866], [144.1736712, 150.22435105, 150.87700865],
                [124.52022402, 128.37772246, 126.18979465], [136.86731207, 167.58542141, 197.19134396],
                [100.7921147, 96.76630824, 73.3046595], [64.54837141, 63.93437044, 55.37530384],
                [118.40495423, 122.40441572, 120.99084545], [159.37921348, 160.05805243, 154.34222846],
                [21.79936809, 33.40047393, 4.82543444], [41.3115, 50.9955, 18.113],
                [105.39016801, 108.99377722, 105.63658992], [162.90212443, 179.72078907, 199.27389985],
                [41.16608997, 68.55305652, 95.37831603], [136.53020134, 115.29817833, 90.46979866],
                [105.40329218, 68.73868313, 57.25514403], [126.52834225, 164.00481283, 199.0802139],
                [120.3922956, 158.33490566, 194.98899371], [124.72755807, 158.87507847, 193.26616447],
                [65.5451895, 75.4271137, 36.10641399], [75.79049034, 89.92273403, 28.5884101],
                [51.26712329, 82.7739726, 112.00293542], [96.43830207, 106.78479763, 111.88351431],
                [172.73444851, 174.16989022, 170.92629378], [109.68316299, 115.71597633, 116.0478752]])

B07 = np.array([[35.55022075, 41.70971302, 47.26379691], [5.34590283, 7.95286439, 1.98368383],
                [213.22107728, 212.1175644, 210.49508197], [248.58068362, 246.91017488, 244.94793323],
                [121.06894357, 155.01442512, 189.68328384], [75.64580726, 97.72715895, 118.5456821],
                [58.2903693, 68.87907314, 27.31788559], [234.26640046, 230.40787222, 226.39589276],
                [84.85313175, 112.1349892, 136.13282937], [150.30047393, 186.32322275, 218.58483412],
                [33.7408323, 46.24680676, 11.12113721], [191.57617348, 195.16936591, 198.93110074],
                [110.35, 121.24642857, 126.20803571], [140.71279373, 86.39791123, 67.65848564],
                [61.36217133, 80.01611535, 100.66157761], [65.69717773, 92.20900076, 117.02135774],
                [20.14285714, 26.34846462, 13.7823765], [128.64655172, 139.10344828, 146.33890086],
                [49.51823119, 62.66252909, 16.12412723], [85.81062124, 98.67735471, 45.04008016],
                [139.57765152, 175.32954545, 206.7260101], [77.42078285, 77.97017707, 69.65703635],
                [48.28785607, 58.70314843, 25.13868066], [116.57762238, 151.58723776, 187.71381119],
                [24.98484848, 35.62962963, 11.04461279], [27.80732177, 23.33526012, 15.55876686],
                [148.12708831, 134.12887828, 110.63066826], [92.15462494, 97.52277957, 97.93741371],
                [15.59797724, 15.64475348, 11.7357775], [34.20644851, 32.83108758, 27.41193455],
                [133.4973638, 159.39015817, 186.20386643], [58.43085106, 73.79388298, 16.80585106],
                [143.54264432, 144.08007449, 138.11880819], [130.01339829, 126.29354446, 114.17174178],
                [117.54252684, 115.91824938, 105.56069364], [129.63451004, 133.84844796, 132.21698113],
                [51.14361446, 49.37156627, 36.41927711], [109.55313837, 144.12196862, 181.25463623],
                [14.45707547, 23.30990566, 4.0759434], [185.88717949, 185.76538462, 181.57564103],
                [31.26196809, 37.02460106, 14.78125, ], [144.38765182, 150.3395749, 150.94129555],
                [124.4720403, 128.37178841, 126.25944584], [137.11750484, 167.68230174, 197.20889749],
                [101.2012873, 96.74370977, 73.70626097], [64.57212318, 63.97893031, 55.38249595],
                [118.40656902, 122.35463826, 120.91389259], [159.55275779, 160.16506795, 154.4008793],
                [21.84956925, 33.35321405, 4.77998675], [41.2171381, 50.99001664, 18.13352745],
                [105.47666492, 108.85317252, 105.37808076], [163.10729881, 179.85152838, 199.32002495],
                [41.14376218, 68.4254386, 95.19590643], [135.91195477, 115.37318255, 90.40387722],
                [105.69550173, 68.25259516, 57.20588235], [126.65223155, 164.15863897, 199.19222271],
                [120.41931217, 158.32936508, 194.9728836], [124.71684945, 158.8664008, 193.28664008],
                [65.75866337, 75.6144802, 36.3595297], [75.74878049, 89.77682927, 28.8097561],
                [50.96543408, 82.62861736, 112.03215434], [96.43429487, 106.74038462, 111.79487179],
                [172.75719982, 174.14089499, 170.79131591], [109.63680494, 115.67034422, 116.00132392]])

B08 = np.array([[35.6620326, 41.9472675, 47.62991371], [5.32159698, 7.88588494, 1.95661742],
                [213.13694268, 211.99124204, 210.31767516], [248.56323381, 246.87151962, 244.92519289],
                [121.02176259, 154.98741007, 189.67625899], [75.53678756, 97.87979275, 118.84352332],
                [58.28248932, 68.78218426, 27.51128737], [234.19870195, 230.32201697, 226.31752371],
                [84.94771242, 112.19607843, 136.1914099], [150.11147812, 186.24360033, 218.58876961],
                [33.7475763, 46.26714542, 11.0021544], [191.58984652, 195.18465171, 199.00354191],
                [110.35258359, 121.34954407, 126.35942249], [141.75726811, 85.90816797, 67.20673743],
                [61.27272727, 80.04727273, 100.77963636], [65.66844563, 92.09873249, 116.81387592],
                [20.04872647, 26.2248062, 13.81284607], [128.61093099, 139.11162575, 146.3955535],
                [49.47136274, 62.67676103, 16.0283081], [86.0929432, 98.94148021, 45.14285714],
                [139.66233766, 175.37337662, 206.70454545], [77.53600945, 78.01692247, 69.58677686],
                [48.37688442, 58.72550251, 25.25251256], [116.55702226, 151.55057559, 187.67766692],
                [25.11752137, 35.73646724, 10.89316239], [27.82371795, 23.43830128, 15.64262821],
                [148.08136615, 134.03917629, 110.49773983], [92.04134886, 97.33801686, 97.83340024],
                [15.58829902, 15.61159263, 11.7156013], [34.36689768, 33.00937627, 27.54300856],
                [133.69325153, 159.49539877, 186.24079755], [58.64334086, 74.01580135, 17.003386, ],
                [143.50960307, 144.08834827, 138.12259923], [130.2539185, 126.31765935, 113.79832811],
                [117.66202346, 115.77859238, 105.18914956], [129.82238571, 133.95822386, 132.20152391],
                [51.28589212, 49.21908714, 36.52033195], [109.58475091, 144.15492102, 181.29191981],
                [14.30815589, 23.2057051, 4.02932905], [185.68773636, 185.69908158, 181.73041599],
                [31.30839002, 37.15986395, 14.94671202], [144.54943625, 150.42150911, 150.94015611],
                [124.40083682, 128.37573222, 126.38870293], [137.23626374, 167.7772612, 197.26627219],
                [101.51820449, 96.80997506, 73.96458853], [64.53628751, 63.92986741, 55.43230984],
                [118.4116095, 122.34602337, 120.86053524], [159.54359862, 160.19204152, 154.50726644],
                [21.78420149, 33.22495707, 4.75329136], [41.18417317, 50.99432221, 18.10574876],
                [105.51708633, 108.74865108, 105.18929856], [163.30110935, 179.99207607, 199.39672478],
                [41.11022632, 68.30259849, 95.00419111], [135.37191249, 115.42907551, 90.31545519],
                [106.02932551, 68.00293255, 57.04252199], [126.72368925, 164.25870647, 199.26980482],
                [120.45531197, 158.34682406, 194.99213041], [124.76473082, 158.89614243, 193.30563798],
                [65.90085745, 75.72347267, 36.56538049], [75.64974619, 89.68020305, 28.93299492],
                [50.76780822, 82.55410959, 112.00753425], [96.37185588, 106.67980965, 111.68456832],
                [172.74894109, 174.10088564, 170.71043512], [109.5968394, 115.65600882, 115.97023153]])

B10 = np.array([[35.6633959, 42.05674061, 47.82977816], [5.2624249, 7.8872201, 1.9609503],
                [213.02999294, 211.900494, 210.24664785], [248.55817378, 246.87893962, 244.9281296],
                [120.97836836, 154.95766378, 189.65698393], [75.57992895, 98.04174067, 119.06483126],
                [58.21147715, 68.70031881, 27.62274176], [234.13735783, 230.25590551, 226.23622047],
                [85.09868421, 112.28453947, 136.32730263], [149.93595342, 186.2147016, 218.59606987],
                [33.75056036, 46.29682997, 10.97246238], [191.58752347, 195.18360108, 199.0210724],
                [110.4192673, 121.39416554, 126.43554953], [142.71534653, 85.44719472, 66.80239274],
                [61.14738203, 80.03684551, 100.82223659], [65.57750583, 92.00990676, 116.65501166],
                [20.06746765, 26.18761553, 13.83271719], [128.57283545, 139.11694707, 146.44480919],
                [49.3891709, 62.66553864, 15.98759165], [86.39328358, 99.14253731, 45.18955224],
                [139.70577105, 175.43093661, 206.71712394], [77.57205982, 77.98844324, 69.52549286],
                [48.36015119, 58.66792657, 25.33531317], [116.53442667, 151.52629439, 187.64760098],
                [25.20120482, 35.86144578, 10.76626506], [27.82792666, 23.51480959, 15.77291961],
                [148.15703573, 134.03837671, 110.44905161], [91.94436468, 97.21715721, 97.83309404],
                [15.54068117, 15.53689688, 11.66556291], [34.47919605, 33.12094499, 27.61953456],
                [133.9877551, 159.59319728, 186.16734694], [58.84292683, 74.18243902, 17.14926829],
                [143.53018081, 144.08511822, 138.09680111], [130.40402194, 126.25411335, 113.55941499],
                [117.68601583, 115.70316623, 104.95382586], [129.95903726, 134.04906272, 132.25549641],
                [51.41985294, 49.21102941, 36.51360294], [109.56168049, 144.14771207, 181.27134065],
                [14.22305229, 23.13020277, 3.99003913], [185.51213926, 185.65185525, 181.91250573],
                [31.31690141, 37.28722334, 15.08249497], [144.68203975, 150.48218973, 150.95650544],
                [124.36832676, 128.38050878, 126.44070226], [137.38120104, 167.83961209, 197.29541216],
                [101.83965517, 96.71551724, 74.17413793], [64.50122624, 63.8908645, 55.48651134],
                [118.4033366, 122.33202486, 120.83087995], [159.53786708, 160.18516229, 154.53724884],
                [21.76895492, 33.15932377, 4.73104508], [41.12854031, 50.99284158, 18.17149082],
                [105.5521859, 108.64040961, 105.02244978], [163.48333333, 180.10416667, 199.45740741],
                [41.04232609, 68.23555392, 94.94037541], [134.81993769, 115.45046729, 90.25545171],
                [106.34012739, 67.58980892, 57.0866242], [126.79494949, 164.35555556, 199.32727273],
                [120.46231884, 158.36956522, 195.01642512], [124.77162505, 158.90336591, 193.31451321],
                [66.00570613, 75.79457917, 36.6723728], [75.6112532, 89.65046888, 29.02387042],
                [50.56803327, 82.47534165, 112.01247772], [96.31940469, 106.64968517, 111.60675444],
                [172.7618394, 174.09814688, 170.68050789], [109.60082045, 115.65257179, 115.94288419]])

B09 = np.array([[35.62982998, 42.09003091, 47.97990726], [5.25705175, 7.84866323, 1.95584989],
                [212.93302469, 211.80771605, 210.18549383], [248.55339806, 246.88401994, 244.92941485],
                [120.949573, 154.9442863, 189.65351769], [75.51789794, 98.22772277, 119.35872049],
                [58.15573384, 68.60689004, 27.78150071], [234.13459314, 230.25530274, 226.24990359],
                [85.13126844, 112.31710914, 136.40560472], [149.74611399, 186.1742228, 218.70336788],
                [33.76635785, 46.31615518, 10.90619572], [191.62046828, 195.21431269, 199.05891239],
                [110.498779, 121.47802198, 126.51770452], [143.39527027, 85.12725225, 66.49286787],
                [61.18444949, 80.09591373, 100.88365494], [65.4913748, 91.94093048, 116.60637742],
                [20.0435133, 26.17163578, 13.82352941], [128.57289003, 139.11216661, 146.4398977],
                [49.30321285, 62.65311245, 15.90060241], [86.59906604, 99.25817211, 45.27951968],
                [139.71923077, 175.46239316, 206.76538462], [77.55632739, 78.02899426, 69.50075506],
                [48.37445991, 58.64234277, 25.4037446], [116.53597166, 151.5137413, 187.62110663],
                [25.28146214, 35.99321149, 10.64334204], [27.86959171, 23.57221207, 15.8031688],
                [148.13411765, 134.02392157, 110.41686275], [91.86574372, 97.10656793, 97.77817128],
                [15.57281553, 15.49810046, 11.60911777], [34.51106331, 33.14781807, 27.69944683],
                [134.17799753, 159.70580964, 186.14833127], [59.0539629, 74.32293423, 17.22681282],
                [143.47725025, 144.05514342, 138.06181998], [130.55016722, 126.32608696, 113.35200669],
                [117.78972521, 115.61111111, 104.74074074], [130.05675228, 134.12096106, 132.28106877],
                [51.51048253, 49.13843594, 36.59467554], [109.55764203, 144.15783218, 181.28904207],
                [14.16044657, 23.08867624, 3.96076555], [185.35242987, 185.58395891, 182.06282102],
                [31.35085355, 37.39757412, 15.18823001], [144.85751892, 150.55018098, 150.92135571],
                [124.31926204, 128.37554722, 126.51000625], [137.45174129, 167.88822554, 197.33930348],
                [102.11978971, 96.80172738, 74.4299662], [64.53825062, 63.91307924, 55.54346038],
                [118.41440139, 122.30277617, 120.75910931], [159.53755482, 160.19435307, 154.56304825],
                [21.75080312, 33.09545663, 4.7117944], [41.10440771, 50.97823691, 18.20275482],
                [105.56692362, 108.59110021, 104.91205326], [163.6171713, 180.18830361, 199.48734965],
                [41.02366864, 68.11406969, 94.79092702], [134.4329438, 115.60100167, 90.22481914],
                [106.5250291, 67.38766007, 57.06635623], [126.86710764, 164.42152736, 199.36921227],
                [120.46153846, 158.37314067, 195.00594985], [124.78291139, 158.9193038, 193.33607595],
                [66.09754029, 75.89228159, 36.74766751], [75.60492905, 89.6034354, 29.01344287],
                [50.39381227, 82.3765076, 111.98164657], [96.28171828, 106.64085914, 111.57492507],
                [172.79391892, 174.04576167, 170.52211302], [109.57684905, 115.6211163, 115.9051416]])

B10 = np.array([[35.65548567, 42.18902865, 48.12403913], [5.25033497, 7.79566771, 1.94685127],
                [212.84238374, 211.73517126, 210.1272626], [248.53287929, 246.85533114, 244.8966651],
                [120.93128849, 154.93116773, 189.65040454], [75.49694086, 98.37933379, 119.62950374],
                [58.09682875, 68.51627907, 27.92262156], [234.1078636, 230.24043145, 226.23173278],
                [85.21784777, 112.46391076, 136.54855643], [149.58782201, 186.12822014, 218.74590164],
                [33.78433957, 46.32639072, 10.8426048], [191.5848537, 195.22409639, 199.09931153],
                [110.48956044, 121.56043956, 126.63736264], [144.05918789, 84.86097729, 66.29490709],
                [61.06908905, 80.0839304, 100.9406346], [65.44969296, 91.88615966, 116.54180444],
                [20.02190813, 26.13144876, 13.7844523], [128.57384106, 139.1281457, 146.4910596],
                [49.23815848, 62.61133245, 15.8247012], [86.78276481, 99.4117295, 45.32854578],
                [139.79528776, 175.50753187, 206.80146775], [77.59388528, 78.06601732, 69.49188312],
                [48.38663793, 58.5862069, 25.45948276], [116.52275509, 151.50027818, 187.61132747],
                [25.38237965, 36.11852861, 10.51544051], [27.90928495, 23.65581644, 15.88847385],
                [148.19539823, 133.99115044, 110.3759292], [91.76577634, 97.00704432, 97.71147637],
                [15.58178368, 15.44364326, 11.53624288], [34.56656518, 33.21754774, 27.77442569],
                [134.29988975, 159.75744212, 186.20396913], [59.23449319, 74.39107413, 17.39334342],
                [143.43722272, 144.04813665, 138.0461402], [130.6749811, 126.30990174, 113.16402116],
                [117.86728061, 115.58667389, 104.41115926], [130.14809936, 134.16390666, 132.30203237],
                [51.54578532, 49.10097029, 36.67798666], [109.56474359, 144.17264957, 181.31431624],
                [14.08264226, 23.04003432, 3.92422076], [185.12743056, 185.49895833, 182.22118056],
                [31.36382114, 37.48414634, 15.30284553], [144.98142141, 150.61633736, 150.93482748],
                [124.27490262, 128.38258208, 126.59599332], [137.62119403, 167.94268657, 197.33641791],
                [102.38326586, 96.73684211, 74.58974359], [64.58766716, 63.95542348, 55.56513125],
                [118.4305447, 122.28303362, 120.70758405], [159.60607557, 160.23437886, 154.56063225],
                [21.72734795, 32.9689698, 4.73810509], [41.08066933, 50.97227772, 18.17782218],
                [105.60018993, 108.5416271, 104.85248496], [163.70747041, 180.2714497, 199.54585799],
                [41.02234138, 68.11230265, 94.77509681], [134.0795569, 115.59718026, 90.18831823],
                [106.74019088, 67.30752916, 56.91834571], [126.94260307, 164.49474535, 199.40662894],
                [120.46050613, 158.37576687, 195.0118865], [124.78953261, 158.92751189, 193.33389309],
                [66.18689415, 75.96355176, 36.78169833], [75.60766182, 89.59313078, 29.01387054],
                [50.25880695, 82.31517144, 111.96195397], [96.2765202, 106.61651132, 111.52463382],
                [172.82217866, 174.01260151, 170.39568748], [109.56567128, 115.59917456, 115.89706239]])

image = mpimg.imread('./mailru.jpg')
data = image.reshape((image.shape[0] * image.shape[1], 3))

ax.set_zlim(0, 255)
plt.xlabel("X")
plt.ylabel("Y")

# ax.scatter(A10.T[0], A10.T[1], A10.T[2], label="10")
# ax.scatter(A30.T[0], A30.T[1], A30.T[2], label="30")
# ax.scatter(data.T[0], data.T[1], data.T[2], label="data")
# ax.scatter(B02.T[0], B02.T[1], B02.T[2])
# plt.savefig("fig02.jpg")
# ax.scatter(B03.T[0], B03.T[1], B03.T[2])
# plt.savefig("fig03.jpg")
# ax.scatter(B04.T[0], B04.T[1], B04.T[2])
# plt.savefig("fig04.jpg")
# ax.scatter(B05.T[0], B05.T[1], B05.T[2])
# plt.savefig("fig05.jpg")
# ax.scatter(B06.T[0], B06.T[1], B06.T[2])
# plt.savefig("fig06.jpg")
# ax.scatter(B07.T[0], B07.T[1], B07.T[2])
# plt.savefig("fig07.jpg")
# ax.scatter(B08.T[0], B08.T[1], B08.T[2])
# plt.savefig("fig08.jpg")
# ax.scatter(B09.T[0], B09.T[1], B09.T[2])
# plt.savefig("fig09.jpg")
# ax.scatter(B10.T[0], B10.T[1], B10.T[2])
# plt.savefig("fig10.jpg")


# plt.legend()
# plt.show()