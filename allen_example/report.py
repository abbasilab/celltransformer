"""Standalone HTML with ALL section maps; ZIP includes PNGs and numeric tables."""
import argparse, base64, html, io, zipfile, shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
from scipy.cluster.hierarchy import linkage,leaves_list
from common import *
from analyze import METHODS
from reference_colors import distinct_references
from scipy.spatial import cKDTree
from matplotlib.collections import LineCollection

COLORS={'celltransformer':'#cf3341','celltransformer_smoothed':'#217ca6','reference':'#8862a9'}

def image64(path):return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()
def esc(value):return html.escape(str(value))
def palette(k):
    import colorsys
    return [colorsys.hsv_to_rgb((i*.61803398875)%1,.58+(i%3)*.12,.75+(i%2)*.2) for i in range(k)]
def save(fig,path):
    fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
def row_order(x):
    return leaves_list(linkage(x,method='average',metric='euclidean')) if len(x)>1 else np.arange(len(x))



def cortex_bounds(xy,structures,divisions):
    mask=np.array([str(v).startswith(('MO','SS')) for v in structures])
    if mask.sum()<100:mask=np.asarray(divisions)=='Isocortex'
    if mask.sum()<100:return None
    return (xy[mask].min(0)-100,xy[mask].max(0)+100)


def reference_boundaries(xy,labels,bounds):
    # Display aid only: a nearest-cell raster, clipped 60 um from any known cell.
    # This does not re-register tissue or enter the comparison metrics.
    lo,hi=bounds;gx=np.arange(lo[0],hi[0]+20,20);gy=np.arange(lo[1],hi[1]+20,20)
    xx,yy=np.meshgrid(gx,gy);points=np.c_[xx.ravel(),yy.ravel()]
    known=labels>=0
    if not known.any():return np.empty((0,2,2))
    distance,near=cKDTree(xy[known]).query(points,k=1)
    grid=labels[known][near].reshape(xx.shape);valid=(distance<=60).reshape(xx.shape)
    edges=[]
    for axis in [0,1]:
        if axis==0:
            a=grid[:-1];b=grid[1:];ok=valid[:-1]&valid[1:];y,x=np.nonzero((a!=b)&ok)
            for j,i in zip(y,x):edges.append([[gx[i]-10,gy[j]+10],[gx[i]+10,gy[j]+10]])
        else:
            a=grid[:,:-1];b=grid[:,1:];ok=valid[:,:-1]&valid[:,1:];y,x=np.nonzero((a!=b)&ok)
            for j,i in zip(y,x):edges.append([[gx[i]+10,gy[j]-10],[gx[i]+10,gy[j]+10]])
    return np.asarray(edges)


def main(run):
    run=Path(run);out=run/'report';figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    cfg=configuration(run);prep=read_json(run/'prepared/summary.json');summary=read_json(run/'celltransformer/summary.json');epochs=pd.DataFrame(read_json(run/'celltransformer/epochs.json'));refs=distinct_references(read_json(run/'analysis/references.json'),read_json(HERE/'reference/allen_structure_graph.json'));metrics=run/'analysis/metrics';scores=pd.read_csv(metrics/'fig2_metrics.tsv',sep='\t')
    fig,ax=plt.subplots(figsize=(7,3));ax.plot(epochs.epoch,epochs.training_nll,'o-',label='Training NLL');ax.plot(epochs.epoch,epochs.validation_nll,'o-',label='Random-center validation NLL');ax.set(xlabel='Completed full epoch',ylabel='Mean ZINB negative log-likelihood');ax.legend();ax.grid(alpha=.2);save(fig,figures/'training.png')
    fig,axes=plt.subplots(1,4,figsize=(17,3.6));columns=['neighborhood_spatial_similarity','proportion_of_discrete_domains','max_domain_CCF_similarity','matched_domain_CCF_similarity'];titles=['Neighborhood spatial\nsimilarity by model','Proportion of discrete\ndomains by model','Max. domain CCF\nsimilarity by model','Matched domain CCF\nsimilarity by model']
    for ax,col,title in zip(axes,columns,titles):
        for method,frame in scores.groupby('method',sort=False):
            if method=='reference' and col in columns[2:]:continue
            frame=frame.sort_values('k');ax.plot(frame.k,frame[col],'o--' if method=='reference' else 'o-',color=COLORS[method],label=frame.method_name.iloc[0])
        ax.set(title=title,xlabel='k',ylabel='Average Pearson correlation' if col in columns[2:] else 'Proportion');ax.grid(alpha=.25)
    axes[-1].legend(loc='upper left',bbox_to_anchor=(1.02,1));fig.tight_layout();save(fig,figures/'fig2_metrics.png')
    meta=pd.read_parquet(run/'prepared/cells.parquet');xy=array(run,'xy_um');sec=array(run,'sections');eligible=array(run,'eligible');vocab=read_json(run/'prepared/vocabulary.json');gallery={};cortex_gallery={};legends={};map_rows=[];structures=meta.parcellation_structure.astype(str).to_numpy();divisions=meta.parcellation_division.astype(str).to_numpy()
    section_ids={s:(int(s.rsplit('.',1)[1]) if '.' in s else s) for s in vocab['section']}
    for level,info in refs.items():
        k=info['k'];pal=palette(k);released=array(run,'reference_'+level);original=read_json(run/'prepared/references.json')[level];refmap={original['ids'].index(i):c for i,c in zip(info['ids'],info['colors'])}
        labels={method:np.load(run/f'analysis/labels/{method}_k{k}.npy',mmap_mode='r') for method in METHODS}
        gallery[level]={};cortex_gallery[level]={};legends[level]={}
        for s,section in enumerate(vocab['section']):
            ids=np.flatnonzero(sec==s);coordinate=xy[ids];mapped=np.array([refmap.get(int(i),'#dddddd') for i in released[ids]])
            fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
            axes[0].scatter(coordinate[:,0],coordinate[:,1],c=mapped,s=.7,linewidths=0,rasterized=True)
            axes[0].set_title('CCF (Allen registration)')
            for ax,(method,display) in zip(axes[1:],METHODS.items()):
                lab=labels[method][ids];ok=lab>=0
                ax.scatter(coordinate[~ok,0],coordinate[~ok,1],c='#dddddd',s=.7,linewidths=0)
                ax.scatter(coordinate[ok,0],coordinate[ok,1],c=np.asarray(pal)[lab[ok]],s=.7,linewidths=0,rasterized=True);ax.set_title(display)
            for ax in axes:ax.set_aspect('equal');ax.set_xlabel('x (µm)')
            axes[0].set_ylabel('y (µm)');axes[0].invert_yaxis()
            fig.suptitle(f'{section} / paper {section_ids.get(section,"?")} · {info["label"]} · k={k} · {len(ids):,} cells',fontsize=12)
            path=figures/f'{section}_{level}.png'
            fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');gallery[level][section]=image64(path)
            bounds=cortex_bounds(coordinate,structures[ids],divisions[ids])
            if bounds is not None:
                lo,hi=bounds;edges=reference_boundaries(coordinate,released[ids],bounds)
                for ax in axes:
                    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
                    if len(edges):ax.add_collection(LineCollection(edges,colors='black',linewidths=.35,alpha=.4))
                fig.suptitle(f'{section} · cortical enlargement · {info["label"]} · k={k}',fontsize=12)
                cp=figures/f'{section}_{level}_cortex.png';save(fig,cp);cortex_gallery[level][section]=image64(cp)
                map_rows.append(dict(section=section,paper_section=section_ids.get(section),reference=level,k=k,image=cp.name,n_cells=len(ids),view='cortex'))
            else:
                plt.close(fig);cortex_gallery[level][section]=None
            entries=[]
            for code in np.unique(released[ids]):
                if code in refmap:
                    name=original['names'][int(code)];count=int((released[ids]==code).sum());color=refmap[code]
                    entries.append(f'<span style="display:inline-block;margin:5px 12px 5px 0"><i style="display:inline-block;width:14px;height:14px;background:{color};border:1px solid #aaa"></i> {esc(name)} ({count:,})</span>')
            legends[level][section]=''.join(entries)

            map_rows.append(dict(section=section,paper_section=section_ids.get(section),reference=level,k=k,image=path.name,n_cells=len(ids),view='whole_section'))
        log(f'Report: embedded {len(vocab["section"])} section maps at {level}, k={k}')
    heatmaps=[];level='substructure';info=refs[level];k=info['k'];refc=pd.read_csv(metrics/f'{level}_reference_cluster_composition.tsv',sep='\t',index_col=0);ro=row_order(refc.to_numpy())
    for method,display in METHODS.items():
        corr=pd.read_csv(metrics/f'{level}_{method}_correlations.tsv',sep='\t',index_col=0);comp=pd.read_csv(metrics/f'{level}_{method}_cluster_composition.tsv',sep='\t',index_col=0);co=row_order(comp.to_numpy())
        fig,axes=plt.subplots(1,3,figsize=(18,max(7,min(18,k*.035))));m=axes[0].imshow(corr.to_numpy()[ro][:,co],vmin=-1,vmax=1,cmap='coolwarm',aspect='auto',interpolation='nearest')
        bright=np.zeros((k,k,4));mask=corr.to_numpy()[ro][:,co]>.9;bright[mask]=[0,1,.1,1];axes[0].imshow(bright,aspect='auto',interpolation='nearest');axes[0].set(title='g · Region-by-domain subclass correlation',xlabel='Discovered domains',ylabel='Reference regions');fig.colorbar(m,ax=axes[0],shrink=.5,label='Pearson r; green > 0.9')
        for ax,table,order,title in [(axes[1],refc,ro,'h · Reference cell-cluster composition'),(axes[2],comp,co,'i · CT2 cell-cluster composition')]:
            im=ax.imshow(table.to_numpy()[order],vmin=0,vmax=1,cmap='magma',aspect='auto',interpolation='nearest');ax.set(title=title,xlabel='Published Allen cell clusters',ylabel='Regions/domains');fig.colorbar(im,ax=ax,shrink=.5,label='Fraction of cells in region')
        fig.suptitle(f'{display} · finest released reference · k={k}',fontsize=14);fig.tight_layout();path=figures/f'composition_{method}.png';save(fig,path);heatmaps.append((display,image64(path)))
    pd.DataFrame(map_rows).to_csv(out/'section_images.tsv',sep='\t',index=False)
    pd.DataFrame([dict(reference=level,k=info['k'],id=i,acronym=name,color=color) for level,info in refs.items() for i,name,color in zip(info['ids'],info['names'],info['colors'])]).to_csv(out/'reference_colors.tsv',sep='\t',index=False)
    pd.DataFrame([dict(k=k,domain=i,color=to_hex(c)) for k in sorted(set(v['k'] for v in refs.values())) for i,c in enumerate(palette(k))]).to_csv(out/'domain_colors.tsv',sep='\t',index=False)

    table=scores[['reference','k','method_name']+columns+['n_evaluation_cells','n_domains_on_evaluation_cells']].to_html(index=False,float_format=lambda x:f'{x:.4f}',na_rep='undefined')
    options=''.join(f'<option value="{esc(s)}">{esc(s)} / paper {esc(section_ids.get(s,"?"))}</option>' for s in vocab['section'])
    levels=''.join(f'<option value="{esc(name)}">{esc(info["label"])} (k={info["k"]})</option>' for name,info in refs.items())
    html_text='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Allen 1 mouse · CellTransformer2</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#21323d;background:#f5f7fa;margin:0}main{max-width:1500px;margin:auto;padding:32px}h1,h2{line-height:1.2}h1{font-size:36px}section{background:white;border:1px solid #dce2e8;border-radius:12px;padding:24px;margin:24px 0}img{max-width:100%;height:auto}pre{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #dde2e8}select,button{font:inherit;padding:8px;margin:5px}.note{background:#fff5d9;padding:15px;border-left:4px solid #cca64b}.scroll{overflow:auto}.small{font-size:13px;color:#536773}.badges{font-size:20px;color:#147783}</style></head><body><main>
'''
    if cfg.get('software_test'):html_text+='<div class="note"><b>SOFTWARE TEST ONLY — synthetic cells and toy-model diagnostics. These are not biological results.</b></div>'
    if cfg.get('software_test'):html_text+='<p><strong>SYNTHETIC SOFTWARE TEST ONLY — not biological results.</strong></p>'
    html_text+=f'<h1>Allen 1 mouse · CellTransformer2</h1><p>Training from scratch · {cfg["epochs"]} full epochs · anatomical reference and CT2 domains</p><p class="badges">{prep["n_cells"]:,} released cells · {prep["n_sections"]} sections · {prep["n_model_genes"]:,} model genes · finest k = {k}</p>'
    html_text+='<div class="note"><b>Exploratory analysis of the Allen 1 mouse using CellTransformer2.</b> This run uses the 20231215 released cell annotations and CCF assignments, joined by cell ID to measured expression, with 50 Blank control probes excluded. The run summary reports the exact input and eligible-center counts; no cells are removed simply to force agreement with a paper total. Training starts from random weights for the requested epoch budget. The random 80/20 center split shares spatial context and is not an independent-animal test. Anatomy is excluded from training. Expression-derived subclass types are model inputs and also enter composition metrics. This uses the CT2 model, objective and optimizer, so it is not a numerical reproduction of the original CT1 model.</div>' 

    html_text+=f'<section><h2>Training diagnostics</h2><img src="{image64(figures/"training.png")}" alt="Training and random-center validation NLL"><div class="scroll">{epochs.to_html(index=False,float_format=lambda x:f"{x:.6f}")}</div><p>Embeddings use the final epoch checkpoint. Lower validation NLL does not automatically imply better anatomical domain recovery.</p></section>'
    html_text+=f'<section><h2>Every section: reference and discovered domains</h2><p>Choose a section and reference resolution. Each dot is a cell. Grey indicates missing reference or an ineligible model center. Every anatomical label has a unique color, including individual cortical layers. A structure keeps its color across sections and resolutions. CT2 colors are consistent across sections within each method and k, and are not matched to the anatomical colors.</p><label>Section <select id="section">{options}</select></label><label>Resolution <select id="level">{levels}</select></label><label>View <select id="view"><option value="full">Whole section</option><option value="cortex">Cortical enlargement (MO/SS where present)</option></select></label><button id="zoom">Toggle full-size view</button><p id="viewnote" class="small"></p><div class="scroll"><img id="map" alt="Anatomical reference and CT2 domains, unsmoothed and smoothed"></div><details><summary>Anatomical labels and cell counts in this section</summary><div id="legend"></div></details><p class="small">Coordinates are released reconstructed x/y coordinates in millimeters, multiplied by 1,000 to obtain micrometers. Every section is included. Cortical layers present in the released CCF labels are shown at the finest resolution. These are registered cell maps; no histology images are synthesized. Cortical crops use motor/somatosensory cortex where at least 100 labelled cells occur, otherwise all Isocortex. Enlarged views include approximate boundaries inferred from nearest-cell reference labels on a 20-µm grid, only within 60 µm of a labelled cell. These display outlines are not the original atlas polygons and do not affect any metric.</p></section>'
    html_text+=f'<section><h2>Figure 2c–f comparison</h2><img src="{image64(figures/"fig2_metrics.png")}" alt="Spatial similarity, discreteness and cell-type composition correlations"><div class="scroll">{table}</div><p>The k values equal the released CCF division, structure and substructure partitions: 25, 354 and 670. Explicit unassigned/residual labels are retained as separate categories. The two CT2 variants are evaluated on the same eligible cells. Sparse regions can produce unstable composition estimates. No other baseline methods were run. Smoothing uses the native CT2 Gaussian function with sigma 12.01 µm and up to 500 same-section neighbors; sigma follows the numerical value in the original paper, whose stated FWHM and sigma are inconsistent. Training uses 10 epochs by default and the CT2 optimizer defaults, rather than the original paper’s 40-epoch CT1 fit.</p></section>' 
    for title,img in heatmaps:html_text+=f'<section><h2>Composition matrices · {esc(title)}</h2><img src="{img}" alt="Region correlations and row-normalized cell-cluster composition"><p>Rows are ordered independently by average-linkage clustering of cell-cluster proportions. Raw tables retain every region name and cell-cluster label; green in the correlation matrix means r &gt; 0.9.</p></section>'
    html_text+='<section><h2>Definitions, provenance and limits</h2><details open><summary>Metric definitions</summary><pre>'+esc(json.dumps(read_json(metrics/'definitions.json'),indent=2))+'</pre></details><details><summary>Preparation and release details</summary><pre>'+esc(json.dumps(prep,indent=2))+'</pre></details><details><summary>Run configuration</summary><pre>'+esc(json.dumps(cfg,indent=2))+'</pre></details><details><summary>Native model provenance</summary><pre>'+esc(json.dumps(summary,indent=2))+'</pre></details><p>The 66 supplied CellTransformer2 files are preserved and SHA256 audited. Adapters handle released cell-ID joins, one-GPU execution, full-epoch sampling, checkpoint resume, and section-safe smoothing. The model, ZINB objective, optimizer, scheduler, Gaussian weight function and KMeans fitting function come from that repository.</p><p><a href="https://www.nature.com/articles/s41467-025-64259-4">Original CellTransformer paper</a> · <a href="https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html">Allen measured MERFISH dataset</a> · <a href="https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/releases/20231215/manifest.json">Pinned 20231215 data manifest</a>. Raw expression and atlas data retain the source terms of use.</p><p>The companion results.zip contains this standalone HTML, all section PNGs, comparison tables, reference dictionaries, configuration, training history, and software versions. Large count arrays and checkpoints remain in the run directory.</p></section>'
    # Keys contain only controlled identifiers; JSON escaping protects embedded data.
    html_text+='<script>const maps='+json.dumps(gallery).replace('</','<\/')+';const cortex='+json.dumps(cortex_gallery).replace('</','<\\/')+';const legends='+json.dumps(legends).replace('</','<\/')+';const s=document.getElementById("section"),l=document.getElementById("level"),m=document.getElementById("map"),v=document.getElementById("view");function show(){const inset=cortex[l.value][s.value];m.src=v.value==="cortex"&&inset?inset:maps[l.value][s.value];document.getElementById("viewnote").textContent=v.value==="cortex"&&!inset?"Fewer than 100 Isocortex-labelled cells in this section; showing the whole section.":"";document.getElementById("legend").innerHTML=legends[l.value][s.value]}l.value="substructure";s.onchange=show;l.onchange=show;v.onchange=show;document.getElementById("zoom").onclick=()=>{m.style.maxWidth=m.style.maxWidth==="none"?"100%":"none"};show();</script></main></body></html>'
    (out/'review.html').write_text(html_text)
    with zipfile.ZipFile(out/'results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in [out/'review.html',out/'section_images.tsv',out/'reference_colors.tsv',out/'domain_colors.tsv',*figures.glob('*.png'),*metrics.glob('*.tsv'),*metrics.glob('*.json'),run/'config.json',run/'environment.json',run/'prepared/summary.json',run/'prepared/genes.json',run/'prepared/gene_symbols.json',run/'prepared/vocabulary.json',run/'prepared/references.json',run/'prepared/gene_selection.tsv',run/'prepared/section_inventory.tsv',run/'prepared/annotation_ids_without_counts.tsv',run/'analysis/clustering.json',run/'analysis/smoothing.json',run/'celltransformer/epochs.json',run/'celltransformer/summary.json',run/'celltransformer/input_schema.json',HERE/'CT2_SOURCE_MANIFEST.json',HERE/'DOWNLOAD_MANIFEST.json']:
            if path.exists():z.write(path,str(path.relative_to(run)) if path.is_relative_to(run) else 'provenance/'+path.name)
        for name in ['ct2_environment.txt','rapids_environment.txt','nvidia_smi.txt','lscpu.txt','uname.txt','os_release.txt','meminfo.txt','ct2_conda_explicit.txt','rapids_conda_explicit.txt','rapids_conda_packages.json']:
            path=Path(cfg['root'])/'logs'/name
            if path.exists():z.write(path,'environment/'+name)
        for path in (HERE/'reference').iterdir():
            if path.is_file():z.write(path,'reference/'+path.name)
    json_write(out/'summary.json',dict(n_section_images=len(map_rows),standalone_html=True,n_reference_resolutions=len(refs),methods=list(METHODS.values())))
    log('STANDALONE HTML WITH SECTION MAPS AND RESULTS ZIP: PASSED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
