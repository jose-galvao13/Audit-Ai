"""
AUDITOR-AI · Deteção de Fraude em Redes · PaySim Graph Analytics
Stack: Python · Pandas · NetworkX · Matplotlib
"""

import os, warnings, json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

INPUT_DIR  = "./data/input"
OUTPUT_DIR = "./data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

C = {
    'bg':'#0d1117','panel':'#161b22','border':'#30363d',
    'text':'#e6edf3','sub':'#8b949e',
    'fraud':'#ff4d4d','faint':'#7a1010',
    'legit':'#58a6ff','laint':'#0d419d',
    'merch':'#f0883e','cycle':'#bc8cff',
    'hub':'#ffa657','green':'#3fb950','yellow':'#d29922',
}

# ─── LOAD / GENERATE ──────────────────────────────────────────────────────────
def load_data():
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith(('.xlsx','.xls','.csv')):
            path = os.path.join(INPUT_DIR, fname)
            print(f"[OK] Dataset: {fname}")
            if fname.lower().endswith('.csv'):
                return pd.read_csv(path)
            return pd.read_excel(path, engine='openpyxl')
    print("[~] Gerando PaySim sintetico...")
    return generate_paysim()


def generate_paysim(n=9000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    def txn(step,ttype,amt,orig,ob_o,nb_o,dest,ob_d,nb_d,fraud=1):
        return dict(step=step,type=ttype,amount=amt,
                    nameOrig=orig,oldbalanceOrg=ob_o,newbalanceOrig=nb_o,
                    nameDest=dest,oldbalanceDest=ob_d,newbalanceDest=nb_d,
                    isFraud=fraud,isFlaggedFraud=0)

    # Ring A – linear mule chain
    chain = [f'C{i:06d}' for i in range(1,6)]
    amt = 50000
    for i in range(len(chain)-1):
        rows.append(txn(10+i,'TRANSFER',amt,chain[i],amt,0,chain[i+1],0,amt))
    rows.append(txn(15,'CASH_OUT',amt,chain[-1],amt,0,'M99001',0,amt))

    # Ring B – star hub
    hub = 'C000010'
    spokes = [f'C{i:06d}' for i in range(11,17)]
    total_hub = 0
    for i,sp in enumerate(spokes):
        sa = int(rng.integers(10000,30000))
        rows.append(txn(20+i,'TRANSFER',sa,sp,sa,0,hub,0,sa))
        total_hub += sa
    rows.append(txn(30,'CASH_OUT',total_hub,hub,total_hub,0,'M99002',0,total_hub))

    # Ring C – cycle
    cycle = ['C000020','C000021','C000022']
    ca = 75000
    for i in range(3):
        rows.append(txn(40+i,'TRANSFER',ca,cycle[i],ca,0,cycle[(i+1)%3],0,ca))

    # Ring D – layered wash
    wash = ['C000030','C000031','C000032']
    wa = 120000
    rows.append(txn(50,'TRANSFER',wa,wash[0],wa,0,wash[1],0,wa))
    rows.append(txn(51,'TRANSFER',wa,wash[1],wa,0,wash[2],0,wa))
    rows.append(txn(52,'CASH_OUT',wa,wash[2],wa,0,'M99003',0,wa))

    n_ring = len(rows)
    n_normal = n - n_ring
    types   = ['PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN']
    weights = [0.35,0.22,0.20,0.13,0.10]
    custs   = [f'C{i:06d}' for i in range(100,100+n_normal)]
    merchs  = [f'M{i:05d}' for i in range(1000,1200)]
    for i in range(n_normal):
        t  = rng.choice(types, p=weights)
        a  = float(rng.lognormal(8,1.5))
        ob = float(rng.uniform(0,a*2))
        nb = max(0,ob-a) if t in ('PAYMENT','TRANSFER','CASH_OUT','DEBIT') else ob+a
        dst = rng.choice(merchs if t in ('PAYMENT','DEBIT') else custs)
        rows.append(dict(step=int(rng.integers(1,744)),type=t,amount=round(a,2),
                         nameOrig=custs[i],oldbalanceOrg=round(ob,2),newbalanceOrig=round(nb,2),
                         nameDest=dst,
                         oldbalanceDest=round(float(rng.uniform(0,1e5)),2),
                         newbalanceDest=round(float(rng.uniform(0,1e5)+a),2),
                         isFraud=0,isFlaggedFraud=0))

    df = pd.DataFrame(rows).sample(frac=1,random_state=42).reset_index(drop=True)
    out = os.path.join(INPUT_DIR,'paysim_synthetic.xlsx')
    df.to_excel(out,index=False)
    print(f"[OK] {len(df):,} transacoes ({df['isFraud'].sum()} fraudes) -> {out}")
    return df


def engineer(df):
    df = df.copy()
    for col in ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']:
        df[col] = pd.to_numeric(df[col],errors='coerce').fillna(0)
    df['drain_orig'] = (df['newbalanceOrig']==0)&(df['oldbalanceOrg']>0)
    df['balance_chg_orig'] = df['newbalanceOrig']-df['oldbalanceOrg']
    df['balance_chg_dest'] = df['newbalanceDest']-df['oldbalanceDest']
    return df


# ─── GRAPH ────────────────────────────────────────────────────────────────────
def build_graph(df, max_rows=20000):
    G = nx.DiGraph()
    for _,r in df.head(max_rows).iterrows():
        o,d = str(r['nameOrig']),str(r['nameDest'])
        fraud  = int(r.get('isFraud',0))
        amount = float(r['amount'])
        for node in (o,d):
            if node not in G:
                G.add_node(node,is_fraud=False,
                           is_cust=node.startswith('C'),
                           is_merch=node.startswith('M'),
                           sent=0.0,recv=0.0,txn_n=0)
        G.nodes[o]['sent']  += amount
        G.nodes[o]['txn_n'] += 1
        G.nodes[d]['recv']  += amount
        G.nodes[d]['txn_n'] += 1
        if fraud:
            G.nodes[o]['is_fraud'] = True
            G.nodes[d]['is_fraud'] = True
        if G.has_edge(o,d):
            G[o][d]['weight'] += amount
            G[o][d]['count']  += 1
            if fraud: G[o][d]['fraud'] = True
        else:
            G.add_edge(o,d,weight=amount,count=1,fraud=bool(fraud),type=str(r['type']))
    return G


def detect_rings(G, df):
    R = {}
    fraud_nodes = {n for n,d in G.nodes(data=True) if d.get('is_fraud')}
    R['fraud_nodes'] = fraud_nodes

    comps = []
    for comp in nx.weakly_connected_components(G):
        fn = [n for n in comp if n in fraud_nodes]
        if fn:
            comps.append({'nodes':list(comp),'size':len(comp),
                          'fraud_n':len(fn),'sub':G.subgraph(comp).copy()})
    comps.sort(key=lambda x:x['fraud_n'],reverse=True)
    R['components'] = comps

    try:
        all_cycles = [c for c in nx.simple_cycles(G) if len(c)>=3]
    except Exception:
        all_cycles = []
    R['cycles'] = all_cycles

    fsg = G.subgraph(fraud_nodes).copy() if fraud_nodes else nx.DiGraph()
    R['hubs'] = sorted(fsg.degree(),key=lambda x:x[1],reverse=True)[:10]
    R['pagerank'] = (sorted(nx.pagerank(fsg,weight='weight').items(),
                            key=lambda x:x[1],reverse=True)[:10]
                    if len(fsg)>1 else [])
    R['drain'] = df[df['drain_orig']&(df['isFraud']==1)]['nameOrig'].value_counts()
    return R


# ─── PLOTS ────────────────────────────────────────────────────────────────────
def _ax(ax, title, xlabel=''):
    ax.set_facecolor(C['panel'])
    ax.set_title(title,color=C['text'],fontsize=11,pad=8)
    if xlabel: ax.set_xlabel(xlabel,color=C['sub'],fontsize=8)
    ax.tick_params(colors=C['sub'],labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(C['border'])


def plot_dashboard(df, G, R):
    fig = plt.figure(figsize=(24,30),facecolor=C['bg'])
    fig.suptitle('AUDITOR-AI  |  Detecao de Fraude em Redes  |  PaySim',
                 color=C['text'],fontsize=20,fontweight='bold',y=0.985)
    gs = GridSpec(5,3,figure=fig,hspace=0.52,wspace=0.35,
                  left=0.06,right=0.97,top=0.965,bottom=0.03)

    n_fraud = int(df['isFraud'].sum())
    n_total = len(df)

    # KPI cards
    kpis = [
        ('Total Transacoes', f"{n_total:,}", C['legit']),
        ('Transacoes Fraude', f"{n_fraud:,}", C['fraud']),
        ('Taxa de Fraude', f"{n_fraud/n_total*100:.2f}%", C['yellow']),
    ]
    for col,(label,val,color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0,col])
        ax.set_facecolor(C['panel'])
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5,0.60,val,ha='center',va='center',transform=ax.transAxes,
                color=color,fontsize=30,fontweight='bold')
        ax.text(0.5,0.22,label,ha='center',va='center',transform=ax.transAxes,
                color=C['sub'],fontsize=12)

    # Type bar
    ax1 = fig.add_subplot(gs[1,0])
    _ax(ax1,'Transacoes por Tipo')
    tc = df.groupby('type')['isFraud'].value_counts().unstack(fill_value=0)
    x = np.arange(len(tc)); w=0.35
    l_v = tc.get(0,pd.Series(0,index=tc.index)).values
    f_v = tc.get(1,pd.Series(0,index=tc.index)).values
    ax1.bar(x-w/2,l_v,w,color=C['legit'],alpha=0.85,label='Legitima')
    ax1.bar(x+w/2,f_v,w,color=C['fraud'],alpha=0.85,label='Fraude')
    ax1.set_xticks(x); ax1.set_xticklabels(tc.index,rotation=20,color=C['sub'],fontsize=8)
    ax1.legend(fontsize=8,labelcolor=C['sub'],facecolor=C['panel'])
    ax1.set_ylabel('Count',color=C['sub'],fontsize=8)

    # Fraud rate
    ax2 = fig.add_subplot(gs[1,1])
    _ax(ax2,'Taxa de Fraude por Tipo (%)')
    rates = df.groupby('type')['isFraud'].mean().sort_values()*100
    cols  = [C['fraud'] if v>5 else C['legit'] for v in rates.values]
    bars  = ax2.barh(rates.index,rates.values,color=cols,alpha=0.85)
    for bar,v in zip(bars,rates.values):
        ax2.text(v+0.2,bar.get_y()+bar.get_height()/2,
                 f'{v:.1f}%',va='center',color=C['text'],fontsize=8)

    # Amount dist
    ax3 = fig.add_subplot(gs[1,2])
    _ax(ax3,'Distribuicao de Montante','Montante (USD)')
    cap = float(df['amount'].quantile(0.99))
    bins = np.linspace(0,cap,50)
    ax3.hist(df[df['isFraud']==0]['amount'].clip(upper=cap),
             bins=bins,color=C['legit'],alpha=0.55,density=True,label='Legitima')
    ax3.hist(df[df['isFraud']==1]['amount'].clip(upper=cap),
             bins=bins,color=C['fraud'],alpha=0.80,density=True,label='Fraude')
    ax3.legend(fontsize=8,labelcolor=C['sub'],facecolor=C['panel'])

    # Fraud over time
    ax4 = fig.add_subplot(gs[2,0])
    _ax(ax4,'Fraudes ao Longo do Tempo','Step')
    ts = df.groupby('step')['isFraud'].sum()
    ax4.fill_between(ts.index,ts.values,color=C['fraud'],alpha=0.4)
    ax4.plot(ts.index,ts.values,color=C['fraud'],linewidth=1)
    ax4.set_ylabel('# Fraudes',color=C['sub'],fontsize=8)

    # Top fraud origins
    ax5 = fig.add_subplot(gs[2,1])
    _ax(ax5,'Top 10 Origens de Fraude')
    top_orig = df[df['isFraud']==1].groupby('nameOrig')['amount'].sum().nlargest(10)
    if not top_orig.empty:
        ax5.barh(range(len(top_orig)),top_orig.values,color=C['hub'],alpha=0.85)
        ax5.set_yticks(range(len(top_orig)))
        ax5.set_yticklabels([n[-8:] for n in top_orig.index],fontsize=8,color=C['sub'])
        ax5.set_xlabel('Volume (USD)',color=C['sub'],fontsize=8)

    # Drain accounts
    ax6 = fig.add_subplot(gs[2,2])
    _ax(ax6,'"Contas Laranja" (Drain Accounts)')
    drain = R['drain'].head(10)
    if not drain.empty:
        ax6.barh(range(len(drain)),drain.values,color=C['cycle'],alpha=0.85)
        ax6.set_yticks(range(len(drain)))
        ax6.set_yticklabels([n[-8:] for n in drain.index],fontsize=8,color=C['sub'])
        ax6.set_xlabel('# Ocorrencias',color=C['sub'],fontsize=8)
    else:
        ax6.text(0.5,0.5,'Sem contas laranja\ndetetadas',ha='center',va='center',
                 color=C['sub'],transform=ax6.transAxes)

    # ── Main network graph ────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3,:])
    ax7.set_facecolor(C['panel'])
    ax7.set_title('Gráfico de Fraude | Anos de Lavagem de Dinheiro',
                  color=C['text'],fontsize=13,pad=10)
    fraud_nodes = R['fraud_nodes']

    focus = set(fraud_nodes)
    for fn in list(fraud_nodes):
        if fn in G:
            focus.update(list(G.predecessors(fn))[:3])
            focus.update(list(G.successors(fn))[:3])
    focus = set(list(focus)[:250])
    SG = G.subgraph(focus).copy()

    if len(SG)>1:
        nc = [C['fraud'] if SG.nodes[n].get('is_fraud') else
              (C['merch'] if SG.nodes[n].get('is_merch') else C['legit'])
              for n in SG.nodes()]
        ns = [280 if SG.nodes[n].get('is_fraud') else
              (140 if SG.nodes[n].get('is_merch') else 60)
              for n in SG.nodes()]
        ec = [C['fraud'] if d.get('fraud') else '#30363d' for _,_,d in SG.edges(data=True)]
        ew = [2.0 if d.get('fraud') else 0.4 for _,_,d in SG.edges(data=True)]

        if R['cycles']:
            ring_flat = list({n for c in R['cycles'] for n in c})
            others    = [n for n in SG.nodes() if n not in ring_flat]
            shells    = [s for s in [ring_flat[:60], others[:180]] if s]
            pos = nx.shell_layout(SG, nlist=shells if len(shells)>1 else None)
        else:
            pos = nx.spring_layout(SG,seed=42,k=2.5,iterations=50)

        nx.draw_networkx_edges(SG,pos,ax=ax7,edge_color=ec,width=ew,alpha=0.75,
                               arrows=True,arrowsize=10,arrowstyle='->',
                               connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_nodes(SG,pos,ax=ax7,node_color=nc,node_size=ns,alpha=0.92)
        fraud_labels = {n:n[-6:] for n in SG.nodes() if SG.nodes[n].get('is_fraud')}
        nx.draw_networkx_labels(SG,pos,labels=fraud_labels,ax=ax7,
                                font_size=6,font_color=C['text'])

        handles = [mpatches.Patch(color=C['fraud'],label='Conta Fraudulenta'),
                   mpatches.Patch(color=C['legit'],label='Conta Legitima'),
                   mpatches.Patch(color=C['merch'],label='Merchant/Destino')]
        ax7.legend(handles=handles,loc='upper right',fontsize=9,
                   labelcolor=C['text'],facecolor=C['panel'],framealpha=0.9)
    else:
        ax7.text(0.5,0.5,'SubGráfico de fraude vazio',ha='center',va='center',
                 color=C['sub'],transform=ax7.transAxes,fontsize=13)
    ax7.axis('off')
    for sp in ax7.spines.values(): sp.set_edgecolor(C['border'])

    # Component sizes
    ax8 = fig.add_subplot(gs[4,0])
    _ax(ax8,'Componentes com Fraude (top 12)')
    comps = R['components'][:12]
    if comps:
        ax8.bar(range(len(comps)),[c['fraud_n'] for c in comps],
                color=C['fraud'],alpha=0.8)
        ax8.set_xticks(range(len(comps)))
        ax8.set_xticklabels([f"C{i+1}\n({c['size']})" for i,c in enumerate(comps)],
                            fontsize=7,rotation=30,color=C['sub'])
        ax8.set_ylabel('Nos Fraudulentos',color=C['sub'],fontsize=8)

    # PageRank
    ax9 = fig.add_subplot(gs[4,1])
    _ax(ax9,'PageRank | Contas Mais Influentes')
    pr = R['pagerank'][:8]
    if pr:
        ax9.barh(range(len(pr)),[v for _,v in pr],color=C['hub'],alpha=0.85)
        ax9.set_yticks(range(len(pr)))
        ax9.set_yticklabels([n[-8:] for n,_ in pr],fontsize=8,color=C['sub'])
        ax9.set_xlabel('PageRank Score',color=C['sub'],fontsize=8)
    else:
        ax9.text(0.5,0.5,'N/A',ha='center',va='center',color=C['sub'],transform=ax9.transAxes)

    # Cycles table
    ax10 = fig.add_subplot(gs[4,2])
    _ax(ax10,f'Ciclos Detetados: {len(R["cycles"])}')
    if R['cycles']:
        txt = ''
        for i,cyc in enumerate(R['cycles'][:6]):
            txt += f"Anel {i+1}: "+' -> '.join(n[-6:] for n in cyc)+' -> (loop)\n\n'
        ax10.text(0.05,0.95,txt,transform=ax10.transAxes,
                  color=C['cycle'],fontsize=8,va='top',fontfamily='monospace')
    else:
        ax10.text(0.5,0.5,'Nenhum ciclo\nencontrado',ha='center',va='center',
                  color=C['sub'],transform=ax10.transAxes)

    out = os.path.join(OUTPUT_DIR,'fraud_graph_dashboard.png')
    plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=C['bg'])
    plt.close()
    print(f"[OK] Dashboard -> {out}")
    return out


def plot_ring_details(G, R):
    cycles = R['cycles']
    fraud_n = R['fraud_nodes']
    if not cycles and not fraud_n:
        print("[!] Sem Anos para detalhar.")
        return None

    n_plots = min(len(cycles),4)+1
    fig,axes = plt.subplots(1,n_plots,figsize=(7*n_plots,7),facecolor=C['bg'])
    if n_plots==1: axes=[axes]
    fig.suptitle('Anos de Fraude Detalhados',color=C['text'],fontsize=16,y=1.02)

    for idx,cyc in enumerate(cycles[:4]):
        ax = axes[idx]
        ax.set_facecolor(C['panel'])
        ax.set_title(f'Anel {idx+1} ({len(cyc)} nos)',color=C['text'],fontsize=11)
        ring_set = set(cyc)
        for node in cyc:
            if node in G:
                ring_set.update(list(G.successors(node))[:3])
                ring_set.update(list(G.predecessors(node))[:3])
        sg = G.subgraph(ring_set).copy()
        pos = nx.circular_layout(sg)
        nc  = [C['fraud'] if sg.nodes[n].get('is_fraud') else
               (C['merch'] if sg.nodes[n].get('is_merch') else C['legit'])
               for n in sg.nodes()]
        ec  = [C['fraud'] if d.get('fraud') else C['border'] for _,_,d in sg.edges(data=True)]
        nx.draw_networkx(sg,pos,ax=ax,node_color=nc,edge_color=ec,
                         node_size=350,font_size=6,font_color=C['text'],
                         labels={n:n[-6:] for n in sg.nodes()},
                         arrows=True,arrowsize=15)
        ax.axis('off')

    # Hub star
    ax_last = axes[-1]
    ax_last.set_facecolor(C['panel'])
    ax_last.set_title('Hub Star (Conta Central)',color=C['text'],fontsize=11)
    hubs = R['hubs']
    if hubs:
        top_hub = hubs[0][0]
        hub_set = {top_hub}
        if top_hub in G:
            hub_set.update(list(G.predecessors(top_hub))[:10])
            hub_set.update(list(G.successors(top_hub))[:5])
        sg2 = G.subgraph(hub_set).copy()
        # Shell layout: hub no centro, spokes na camada exterior
        spokes = [n for n in sg2.nodes() if n != top_hub]
        pos2 = nx.shell_layout(sg2, nlist=[spokes, [top_hub]])
        nc2 = [C['hub'] if n==top_hub else
               (C['fraud'] if sg2.nodes[n].get('is_fraud') else C['legit'])
               for n in sg2.nodes()]
        ns2 = [700 if n==top_hub else 380 for n in sg2.nodes()]
        nx.draw_networkx_edges(sg2, pos2, ax=ax_last,
                               edge_color=C['fraud'], width=2.0, alpha=0.8,
                               arrows=True, arrowsize=18, arrowstyle='->',
                               connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_nodes(sg2, pos2, ax=ax_last,
                               node_color=nc2, node_size=ns2, alpha=0.95)
        nx.draw_networkx_labels(sg2, pos2, ax=ax_last,
                                labels={n: n[-6:] for n in sg2.nodes()},
                                font_size=7, font_color=C['text'])
    ax_last.axis('off')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR,'fraud_rings_detail.png')
    plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=C['bg'])
    plt.close()
    print(f"[OK] Detalhe dos Anos -> {out}")
    return out


def export_reports(df, G, R):
    # Fraud CSV
    fraud_df = df[df['isFraud']==1].copy()
    fraud_df['degree_orig'] = fraud_df['nameOrig'].apply(lambda n: G.degree(n) if n in G else 0)
    fraud_df['degree_dest'] = fraud_df['nameDest'].apply(lambda n: G.degree(n) if n in G else 0)
    p1 = os.path.join(OUTPUT_DIR,'fraud_transactions.csv')
    fraud_df.to_csv(p1,index=False)
    print(f"[OK] Transacoes fraudulentas -> {p1}")

    # Node metrics CSV
    node_rows = []
    for n,d in G.nodes(data=True):
        if d.get('is_fraud'):
            node_rows.append({'account':n,'in_degree':G.in_degree(n),
                              'out_degree':G.out_degree(n),
                              'total_sent':round(d['sent'],2),
                              'total_recv':round(d['recv'],2),
                              'txn_count':d['txn_n'],
                              'type':'merchant' if d['is_merch'] else 'customer'})
    p2 = os.path.join(OUTPUT_DIR,'fraud_network_nodes.csv')
    pd.DataFrame(node_rows).sort_values('in_degree',ascending=False).to_csv(p2,index=False)
    print(f"[OK] Nos da rede -> {p2}")

    # Ring summary JSON
    ring_summary = {
        'total_fraud_nodes':  len(R['fraud_nodes']),
        'fraud_components':   len(R['components']),
        'cycles_detected':    len(R['cycles']),
        'cycles':             [list(c) for c in R['cycles'][:20]],
        'top_hubs':           [(n,int(d)) for n,d in R['hubs'][:10]],
        'top_pagerank':       [(n,round(s,6)) for n,s in R['pagerank'][:10]],
        'drain_accounts':     R['drain'].head(10).to_dict(),
    }
    p3 = os.path.join(OUTPUT_DIR,'fraud_ring_summary.json')
    with open(p3,'w') as f: json.dump(ring_summary,f,indent=2)
    print(f"[OK] Ring summary JSON -> {p3}")

    # Neo4j Cypher
    lines = ["// Neo4j Cypher – Fraud Ring Import\n",
             "// Colar no Neo4j Browser ou Aura\n\n"]
    for _,row in fraud_df.head(300).iterrows():
        lines.append(
            f"MERGE (a:Account {{id:'{row['nameOrig']}'}}) "
            f"MERGE (b:Account {{id:'{row['nameDest']}'}}) "
            f"CREATE (a)-[:TRANSACTION {{amount:{row['amount']},"
            f"type:'{row['type']}',step:{row['step']},fraud:true}}]->(b);\n")
    for cyc in R['cycles']:
        for i in range(len(cyc)):
            src,dst = cyc[i],cyc[(i+1)%len(cyc)]
            lines.append(
                f"MERGE (a:Account {{id:'{src}'}}) "
                f"MERGE (b:Account {{id:'{dst}'}}) "
                f"MERGE (a)-[:IN_RING {{ring_id:{R['cycles'].index(cyc)}}}]->(b);\n")
    p4 = os.path.join(OUTPUT_DIR,'neo4j_import.cypher')
    with open(p4,'w') as f: f.writelines(lines)
    print(f"[OK] Neo4j Cypher -> {p4}")

    return p1,p2,p3,p4


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*65)
    print("  AUDITOR-AI  |  PaySim Graph Fraud Detection")
    print("="*65+"\n")

    df = load_data()
    df = engineer(df)
    print(f"[i] {len(df):,} transacoes | {df['isFraud'].sum()} fraudes "
          f"({df['isFraud'].mean()*100:.2f}%)\n")

    print("[~] Construindo Gráfico...")
    G = build_graph(df)
    print(f"[i] {G.number_of_nodes():,} nos | {G.number_of_edges():,} arestas\n")

    print("[~] Detetando Anos de fraude...")
    R = detect_rings(G, df)
    print(f"[i] Nos fraudulentos : {len(R['fraud_nodes'])}")
    print(f"[i] Componentes      : {len(R['components'])}")
    print(f"[i] Ciclos (Anos)    : {len(R['cycles'])}")
    for i,c in enumerate(R['cycles'][:5]):
        print(f"    Anel {i+1}: {' -> '.join(n[-8:] for n in c)} -> (loop)")
    print()

    print("[~] Gerando dashboard...")
    plot_dashboard(df, G, R)

    print("[~] Gerando detalhe dos Anos...")
    plot_ring_details(G, R)

    print("[~] Exportando relatorios...")
    export_reports(df, G, R)

    print("\n"+"="*65)
    print("  CONCLUIDO  |  Outputs em: "+OUTPUT_DIR)
    print("="*65+"\n")