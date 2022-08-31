
from matplotlib import pyplot as plt

def plotDataVsModels(series:list, x_values:list=None, show:bool=False, sort_index:int = None):

    # TODO add label handling

    assert len(series) > 0
    assert all(len(i) == len(series[0]) for i in series)

    if x_values is not None:
        assert len(x_values) == len(series[0])
        xValues = x_values
    else:
        xValues = range(len(series[0]))

    for serie in series:
        serie_toplot = serie
        if sort_index is not None:
            serie_toplot = [x for _, x in sorted(zip(series[sort_index], serie))]
        plt.plot(xValues, serie_toplot)
    
    if show:
        plt.show()

def plot2seriesVsModels(s1, s2, show:bool=False, sort_index:int = None):
    assert len(s1) == len(s2)

    xValues = range(len(s1))

    sort_reference = None
    if sort_index == 0:
        sort_reference = s1
    elif sort_index == 1:
        sort_reference = s2
    else:
        exit(1)

    s1_sorted = [x for _, x in sorted(zip(sort_reference, s1))]
    s2_sorted = [x for _, x in sorted(zip(sort_reference, s2))]

    ratio = 0.5
    def fix_ratio(axS, axT):
        x_left, x_right = axS.get_xlim()
        y_low, y_high = axS.get_ylim()
        axT.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

    from mpl_toolkits.axes_grid1 import host_subplot
    ax1 = host_subplot(111, adjustable='box')
    ax1.plot(xValues, s1_sorted)    
    ax2 = plt.twinx()
    ax2.plot(xValues, s2_sorted)
    fix_ratio(ax1, ax1)
    fix_ratio(ax2, ax2)

    if show:
        plt.show()
