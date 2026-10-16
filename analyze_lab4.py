"""Reproduce Lab 4 experiments 1-3 from the untouched input.tex notebook.

Run with Python, NumPy, SciPy, and Matplotlib. No measurements are generated.
Each original ruler position has u = 0.05 cm, confirmed by the authors.
Two independent bench positions give u_L = sqrt(2) u. The mean magnitude
of opposite transverse coordinates is (x_right-x_left)/2, so the uncertain
central reference cancels and u_y = u/sqrt(2). Left/right half differences
larger than this scale are retained as an extra conservative edge scale.

Each anode setting has an estimated 0.01 V control uncertainty, confirmed
by the authors. As explicitly requested, each stopping fit includes EVERY
recorded point with I > 0, including positive anode voltages. Fit I=m*Va+q
and take the signed zero crossing -q/m. Stopping voltage is its magnitude.
Use voltage and current errors in a generalized orthogonal regression.
Multiply its absolute covariance by max(1, chi_squared/(N-2)) to reflect
the observed scatter around the straight line. This does not make the
visibly curved current characteristic linear. Curvature and current offset
remain systematic errors beyond these conditional regression uncertainties.
"""
from pathlib import Path
import csv
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / 'figures'
DATA = ROOT / 'data'
C = 299792458.0
RULER_U = 0.05
DISTANCE_U = math.sqrt(2) * RULER_U
PAIRED_U = RULER_U / math.sqrt(2)
VOLTAGE_U = 0.01
COLORS = {'Red': 'tab:red', 'Green': 'tab:green', 'Blue': 'tab:blue'}


def save_figure(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES / f'{name}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def write_csv(name, header, rows):
    with (DATA / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def wavelength(y, length, spacing):
    return spacing * y / np.hypot(y, length)


def fit_line_with_covariance(coordinate, covariance, response, response_u):
    """Generalized orthogonal fit with a full coordinate covariance.

    The inverse Jacobian information gives absolute covariance. Any
    residual scaling is applied explicitly by the caller. For the final
    three-point energy fit, do not shrink errors using its small residual.
    """
    cholesky = np.linalg.cholesky(covariance)
    def residual(parameters):
        slope, intercept = parameters[:2]
        latent = parameters[2:]
        return np.r_[np.linalg.solve(cholesky, latent-coordinate),
                     (slope*latent+intercept-response)/response_u]
    initial = np.r_[np.polyfit(coordinate, response, 1), coordinate]
    fit = least_squares(residual, initial, xtol=1e-13, ftol=1e-13, gtol=1e-13)
    full_covariance = np.linalg.inv(fit.jac.T @ fit.jac)
    return fit.x[:2], full_covariance[:2, :2], float(fit.fun @ fit.fun)


def fit_stopping_line(voltage, current, current_u):
    """Use all and only I>0 readings, with no voltage-window selection."""
    selected = current > 0
    x, y, uy = voltage[selected], current[selected], current_u[selected]
    beta, absolute_covariance, chi2 = fit_line_with_covariance(
        x, np.eye(len(x))*VOLTAGE_U**2, y, uy)
    slope, intercept = beta
    degrees = len(x)-2
    reduced = chi2/degrees
    scale = max(1.,reduced)
    covariance = absolute_covariance*scale
    zero = -intercept/slope
    gradient = np.array([intercept/slope**2,-1/slope])
    zero_u = math.sqrt(gradient @ covariance @ gradient)
    residual_current = y-(slope*x+intercept)
    # Vertical residuals at the measured coordinates. Include uncertainty
    # in both axes, with the fitted line held fixed. Do not divide these
    # by the enlarged regression covariance, which would hide curvature.
    residual_scale = np.hypot(uy,slope*VOLTAGE_U)
    return {
        'selection_rule': 'Every recorded point with current_nA > 0',
        'selected_indices': np.flatnonzero(selected).tolist(),
        'selected_voltage_V': x.tolist(), 'selected_current_nA': y.tolist(),
        'point_count': len(x), 'slope_nA_per_V': float(slope),
        'intercept_nA': float(intercept),
        'signed_zero_crossing_V': float(zero), 'stopping_magnitude_V': float(abs(zero)),
        'stopping_u_V': float(zero_u),
        'slope_intercept_covariance': covariance.tolist(),
        'absolute_slope_intercept_covariance': absolute_covariance.tolist(),
        'weighted_residual_sum': chi2, 'degrees_of_freedom': degrees,
        'reduced_weighted_residual': reduced, 'covariance_scale': scale,
        'residual_current_nA': residual_current.tolist(),
        'residual_scale_nA': residual_scale.tolist(),
        'scaled_residuals': (residual_current/residual_scale).tolist(),
    }


def run_analysis():
    FIGURES.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    length1 = 42.95 - 18.73
    length2 = 42.96 - 19.08
    reference_nm = 634.6
    y1 = (8.30 + 8.30) / 2
    spacing = reference_nm * math.hypot(y1, length1) / y1
    spacing_u = spacing * math.hypot(
        length1**2 / (y1*(length1**2+y1**2)) * PAIRED_U,
        length1 / (length1**2+y1**2) * DISTANCE_U)
    density = 1e7 / spacing
    density_u = density * spacing_u / spacing
    order_positions = np.array([-22.0, -8.30, 0, 8.30, 19.9])
    angles = np.degrees(np.arctan2(order_positions, length1))
    displacement_u = math.sqrt(2) * RULER_U
    angle_u = np.degrees(np.hypot(length1*displacement_u,
                                  order_positions*DISTANCE_U)
                         / (length1**2 + order_positions**2))
    third_angle = math.asin(3*reference_nm/spacing)
    third_position = length1 * math.tan(third_angle)
    write_csv('grating_positions.csv',
              ['order','side','displacement_cm','angle_deg','reading_angle_u_deg'],
              [[m, side, y, a, u] for m, side, y, a, u in zip(
                  [2,1,0,1,2], ['Left','Left','Center','Right','Right'],
                  order_positions, angles, angle_u)])

    # Rows are red, green, blue. Columns are near left/right, far left/right.
    names = ['Red', 'Green', 'Blue']
    edges = np.array([[7.20,7.23,8.05,8.02],
                      [6.01,6.07,7.70,7.53],
                      [5.42,5.42,6.25,6.40]])
    paired = edges.reshape(3,2,2).mean(axis=2)
    half_difference = np.abs(np.diff(edges.reshape(3,2,2),axis=2)[:,:,0])/2
    paired_error = np.maximum(PAIRED_U, half_difference)
    wavelengths = wavelength(paired, length2, spacing)
    grad_y = spacing*length2**2 / (length2**2+paired**2)**1.5
    grad_length = -spacing*paired*length2 / (length2**2+paired**2)**1.5
    wave_u = np.sqrt((wavelengths*spacing_u/spacing)**2
                     +(grad_y*paired_error)**2
                     +(grad_length*DISTANCE_U)**2)
    midpoint = wavelengths.mean(axis=1)
    bandwidth = wavelengths[:,1]-wavelengths[:,0]
    write_csv('led_edges.csv',
              ['LED','near_left_cm','near_right_cm','far_left_cm','far_right_cm',
               'lambda_min_nm','lambda_min_u_nm','lambda_max_nm','lambda_max_u_nm'],
              [[name,*raw,*[v for pair in zip(w,uw) for v in pair]]
               for name,raw,w,uw in zip(names,edges,wavelengths,wave_u)])

    voltage = np.arange(-1000,401,200,dtype=float)/1000
    currents = np.array([[-5,0,0,0,40,578,1446,2227],
                         [-16,-10,21,338,1294,2730,4280,5800],
                         [-50,110,794,2212,4292,6842,9430,12030]])
    current_u = np.array([[5,5,6,6,5,6,6,8],
                          [4,5,5,5,8,8,8,8],
                          [2,2,3,3,5,5,6,10]])
    write_csv('led_current.csv',
              ['LED','power_setting','anode_voltage_V','anode_voltage_u_V','current_nA','current_u_nA'],
              [[name,'L040',v,VOLTAGE_U,i,u] for name,series,errors in zip(names,currents,current_u)
               for v,i,u in zip(voltage,series,errors)])
    stopping_fits = [fit_stopping_line(voltage,current,error)
                     for current,error in zip(currents,current_u)]
    energy = np.array([fit['stopping_magnitude_V'] for fit in stopping_fits])
    energy_u = np.array([fit['stopping_u_V'] for fit in stopping_fits])
    write_csv('stopping_fits.csv',
              ['LED','positive_point_count','fit_voltage_min_V','fit_voltage_max_V',
               'slope_nA_per_V','intercept_nA','signed_zero_crossing_V',
               'stopping_magnitude_V','stopping_u_V','reduced_weighted_residual'],
              [[name,fit['point_count'],min(fit['selected_voltage_V']),
                max(fit['selected_voltage_V']),fit['slope_nA_per_V'],fit['intercept_nA'],
                fit['signed_zero_crossing_V'],fit['stopping_magnitude_V'],
                fit['stopping_u_V'],fit['reduced_weighted_residual']]
               for name,fit in zip(names,stopping_fits)])
    write_csv('current_residuals.csv',
              ['LED','anode_voltage_V','current_residual_nA','combined_reading_scale_nA',
               'scaled_residual'],
              [[name,x,residual,scale,scaled]
               for name,fit in zip(names,stopping_fits)
               for x,residual,scale,scaled in zip(
                   fit['selected_voltage_V'],fit['residual_current_nA'],
                   fit['residual_scale_nA'],fit['scaled_residuals'])])

    # A visual near edge is the best available proxy for the high-frequency
    # end of the source, not a claim of a sharp physical spectral cutoff.
    near = paired[:,0]
    frequency = C / (wavelengths[:,0]*1e-9) / 1e14
    derivative_d = -frequency/spacing
    derivative_l = frequency*length2/(length2**2+near**2)
    derivative_y = -frequency*length2**2/(near*(length2**2+near**2))
    frequency_cov = (np.outer(derivative_d,derivative_d)*spacing_u**2
                     +np.outer(derivative_l,derivative_l)*DISTANCE_U**2
                     +np.diag((derivative_y*paired_error[:,0])**2))
    beta, beta_cov, chi2 = fit_line_with_covariance(
        frequency, frequency_cov, energy, energy_u)
    slope, intercept = beta
    h = slope*1e-14
    h_u = math.sqrt(beta_cov[0,0])*1e-14
    phi = -intercept
    phi_u = math.sqrt(beta_cov[1,1])
    cutoff = -intercept/slope*1e14
    cutoff_gradient = np.array([intercept/slope**2,-1/slope])*1e14
    cutoff_u = math.sqrt(cutoff_gradient @ beta_cov @ cutoff_gradient)
    energy_residual = energy-(slope*frequency+intercept)
    # Marginal vertical scales include frequency uncertainty. The full
    # frequency covariance remains in the fit. Bars are +/- one input
    # scale with the fitted line fixed, not fitted prediction intervals.
    energy_residual_scale = np.sqrt(energy_u**2+slope**2*np.diag(frequency_cov))
    energy_scaled_residual = energy_residual/energy_residual_scale
    write_csv('energy_residuals.csv',
              ['LED','frequency_1e14_Hz','energy_residual_eV','combined_input_scale_eV',
               'scaled_residual'],
              zip(names,frequency,energy_residual,energy_residual_scale,energy_scaled_residual))
    summary = {
        'ruler_reading_u_cm': RULER_U, 'separation_u_cm': DISTANCE_U,
        'paired_displacement_u_cm': PAIRED_U,
        'grating': {'distance_cm':length1,'spacing_nm':spacing,'spacing_u_nm':spacing_u,
                    'slits_per_cm':density,'slits_per_cm_u':density_u,
                    'third_order_angle_deg':math.degrees(third_angle),
                    'third_order_position_cm':third_position},
        'spectrum': {'distance_cm':length2,'LED_order':names,
                     'mean_edges_cm':paired.tolist(),'mean_edge_scales_cm':paired_error.tolist(),
                     'wavelength_edges_nm':wavelengths.tolist(),
                     'edge_reading_u_nm':wave_u.tolist(),'midpoints_nm':midpoint.tolist(),
                     'bandwidths_nm':bandwidth.tolist()},
        'stopping': {'magnitudes_V':energy.tolist(),'estimated_u_V':energy_u.tolist(),
                     'voltage_setting_u_V':VOLTAGE_U,
                     'method':'Linear zero-current intercept using every I > 0 point',
                     'fits':dict(zip(names,stopping_fits)),
                     'blue_zero_plateau_demonstrated':False},
        'physical_fit': {'frequency_1e14Hz':frequency.tolist(),
                         'frequency_covariance':frequency_cov.tolist(),
                         'slope_eV_per_1e14Hz':slope,'intercept_eV':intercept,
                         'slope_intercept_covariance':beta_cov.tolist(),
                         'h_eVs':h,'h_u_eVs':h_u,'h_Js':h*1.602176634e-19,
                         'h_u_Js':h_u*1.602176634e-19,
                         'work_function_eV':phi,'work_function_u_eV':phi_u,
                         'cutoff_Hz':cutoff,'cutoff_u_Hz':cutoff_u,
                         'weighted_residual_sum':chi2,'degrees_of_freedom':1,
                         'energy_residual_eV':energy_residual.tolist(),
                         'energy_residual_scale_eV':energy_residual_scale.tolist(),
                         'scaled_residuals':energy_scaled_residual.tolist()}}
    (ROOT/'analysis_results.json').write_text(json.dumps(summary,indent=2)+'\n')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    fig, (ax,pat) = plt.subplots(1,2,figsize=(8.3,2.0),
                                gridspec_kw={'width_ratios':[1,1.5]},layout='constrained')
    ax.plot([0,0],[-9,9],color='0.2',lw=2)
    ax.plot([length1,length1],[-12,12],color='0.2',lw=2)
    for y in [-8.3,0,8.3]:
        ax.plot([0,length1],[0,y],color='tab:red',lw=1.3)
    ax.text(0,-11,'Grating',ha='center')
    ax.text(length1,-14,'Screen',ha='center')
    ax.annotate('',(length1,11),(0,11),arrowprops={'arrowstyle':'<->'})
    ax.text(length1/2,12,'L = 24.22 cm',ha='center')
    ax.text(length1+1,4,'y',color='tab:red')
    ax.text(9,1.2,r'$\theta$')
    ax.set(xlim=(-5,31),ylim=(-16,16),aspect='equal')
    ax.axis('off')
    pat.axhline(0,color='0.7',lw=1)
    for m,y in zip([-2,-1,0,1,2],order_positions):
        pat.scatter(y,0,s=25 if abs(m)==2 else 85,color='tab:red',
                    alpha=.35 if abs(m)==2 else 1)
        pat.text(y,0.29,f'm = {m}',ha='center')
        pat.text(y,-.37,f'{y:g}',ha='center')
    pat.text(0,-.8,'Screen displacement / cm',ha='center')
    pat.set(xlim=(-26,24),ylim=(-1.1,.8))
    pat.axis('off')
    save_figure(fig,'grating_geometry')

    # A compact sketch pairs with the new setup photograph in the report.
    fig,pat = plt.subplots(figsize=(4.2,1.6),layout='constrained')
    pat.axhline(0,color='0.7',lw=1)
    for m,y in zip([-2,-1,0,1,2],order_positions):
        pat.scatter(y,0,s=25 if abs(m)==2 else 85,color='tab:red',
                    alpha=.35 if abs(m)==2 else 1)
        pat.text(y,.28,f'm = {m}',ha='center')
        pat.text(y,-.35,f'{y:g}',ha='center')
    pat.text(0,-.8,'Screen displacement / cm',ha='center')
    pat.set(xlim=(-26,24),ylim=(-1.1,.8))
    pat.axis('off')
    save_figure(fig,'grating_pattern')

    fig,(ax,pattern) = plt.subplots(1,2,figsize=(8.2,2.1),layout='constrained',
                                   gridspec_kw={'width_ratios':[1.35,1]})
    nominal = [618,533,462]
    for row,(name,band,spec) in enumerate(zip(names,wavelengths,nominal)):
        ax.plot(band,[row,row],lw=6,color=COLORS[name],alpha=.65,solid_capstyle='butt')
        ax.errorbar(band,[row,row],xerr=wave_u[row],fmt='none',
                    ecolor=COLORS[name],capsize=4)
        ax.errorbar(spec,row+.17,xerr=10,fmt='x',color='0.25',capsize=3)
        for sign in [-1,1]:
            pattern.plot(sign*paired[row],[row,row],lw=6,color=COLORS[name],
                         alpha=.65,solid_capstyle='butt')
        pattern.scatter(0,row,s=50,color=COLORS[name],zorder=3)
    ax.plot([],[],lw=6,color='0.6',alpha=.65,label='Measured visible band')
    ax.plot([],[],'x',color='0.25',label='Manufacturer value and range')
    ax.set(yticks=range(3),yticklabels=names,xlim=(410,650),ylim=(-.35,2.55),
           xlabel='Wavelength / nm')
    ax.invert_yaxis()
    handles,labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',fontsize=8,frameon=False,ncol=2)
    pattern.set(xlim=(-10,10),xticks=[-8,0,8],yticks=range(3),yticklabels=[],
                ylim=(-.35,2.55),xlabel='Screen displacement / cm')
    pattern.invert_yaxis()
    pattern.axvline(0,color='0.8',lw=.8,zorder=1)
    pattern.tick_params(axis='y',length=0)
    pattern.grid(axis='x',alpha=.2)
    ax.grid(axis='x',alpha=.2)
    save_figure(fig,'led_spectra')

    fig,(ax,residual_ax) = plt.subplots(2,1,figsize=(7.8,3.4),sharex=True,
                                       layout='constrained',
                                       gridspec_kw={'height_ratios':[2.1,1]})
    for row,name in enumerate(names):
        ax.errorbar(voltage,currents[row]/1000,xerr=VOLTAGE_U,yerr=current_u[row]/1000,
                    fmt='o-',ms=4,lw=1,capsize=2,color=COLORS[name],label=name)
        fit = stopping_fits[row]
        zero = fit['signed_zero_crossing_V']
        line_x = np.linspace(min(zero,min(fit['selected_voltage_V']))-.03,.42,200)
        line_i = fit['slope_nA_per_V']*line_x+fit['intercept_nA']
        ax.plot(line_x,line_i/1000,'--',lw=1.4,color=COLORS[name])
        ax.errorbar(zero,0,xerr=fit['stopping_u_V'],fmt='x',ms=6,
                    capsize=3,color=COLORS[name],zorder=5)
        residual_ax.errorbar(fit['selected_voltage_V'],fit['scaled_residuals'],
                             yerr=np.ones(fit['point_count']),fmt='o',ms=4,
                             capsize=2,color=COLORS[name])
    ax.set(ylabel=r'Current / $\mu$A',xlim=(-1.05,.45))
    ax.legend(frameon=False,ncol=3,loc='upper left')
    ax.axhline(0,color='0.65',lw=.8)
    residual_ax.axhline(0,color='0.4',lw=.8)
    residual_ax.set(xlabel='Anode voltage / V',ylabel='Scaled residual')
    for plot in (ax,residual_ax): plot.grid(alpha=.2)
    save_figure(fig,'led_current')

    fig,(physical,residual_ax) = plt.subplots(2,1,figsize=(7.8,3.4),sharex=True,
                                             layout='constrained',
                                             gridspec_kw={'height_ratios':[2.1,1]})
    for row,name in enumerate(names):
        physical.errorbar(frequency[row],energy[row],
                          xerr=math.sqrt(frequency_cov[row,row]),yerr=energy_u[row],
                          fmt='o',color=COLORS[name],capsize=3,zorder=3,label=name)
        # Pale segment shows the full visible spectrum, not a second error bar.
        low_f=C/(wavelengths[row,1]*1e-9)/1e14
        physical.plot([low_f,frequency[row]],[energy[row]]*2,
                      lw=4,color=COLORS[name],alpha=.18,zorder=1)
        residual_ax.errorbar(frequency[row],energy_scaled_residual[row],
                             yerr=1,fmt='o',color=COLORS[name],ms=4,capsize=3)
    freq_line=np.linspace(3.85,7.15,150)
    physical.plot(freq_line,slope*freq_line+intercept,color='0.3',lw=1.2)
    physical.axhline(0,color='0.65',lw=.8)
    physical.set(ylim=(-.10,.86),ylabel=r'$K_{\max}$ / eV',xlim=(3.85,7.15))
    physical.legend(frameon=False,ncol=3,loc='upper left')
    residual_ax.axhline(0,color='0.4',lw=.8)
    residual_ax.set(xlabel=r'Near edge frequency / $10^{14}$ Hz',
                    ylabel='Scaled residual',ylim=(-1.5,1.5),yticks=[-1,0,1])
    for plot in (physical,residual_ax):
        plot.grid(alpha=.2)
    save_figure(fig,'photoelectric_fit')

    # The notebook executes this single maintained source, so equations and
    # figures remain consistent with the script submitted alongside the report.
    notebook={'cells':[
        {'cell_type':'markdown','id':'lab4-overview','metadata':{},'source':[
            '# Lab 4 experiments 1 to 3\n',
            'Run all cells to reproduce the analysis and figures. Raw readings are in data/. '
            'See analyze_lab4.py for uncertainty assumptions and analysis_results.json for full precision.']},
        {'cell_type':'code','id':'lab4-analysis','execution_count':None,'metadata':{},'outputs':[],
         'source':['from pathlib import Path\n','import runpy\n',
                   'root = Path.cwd()\n',
                   'if not (root / "analyze_lab4.py").exists():\n',
                   '    root = root / "lab4"\n',
                   'results = runpy.run_path(str(root / "analyze_lab4.py"))["run_analysis"]()\n',
                   'results["physical_fit"]']}],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                    'language_info':{'name':'python','version':'3'}},
        'nbformat':4,'nbformat_minor':5}
    (ROOT/'lab4_analysis.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    result = run_analysis()
    print(json.dumps({'grating':result['grating'],
                      'spectrum':result['spectrum'],
                      'physical_fit':result['physical_fit']},indent=2))
