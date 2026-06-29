"""
Generatore di Expert Advisor MQL5 (.mq5) - Modulo 6.

Versione MQL5 (API a handle + CopyBuffer, classe CTrade) dello stesso EA
generato per MQL4: ricalcola le feature native, standardizza, trova il cluster
piu' vicino col KMeans embeddato e gestisce il trade con SL/TP/BE/trailing in
ATR e rischio percentuale.

NB: genera testo MQL5; la compilazione va fatta in MetaEditor.
"""

from __future__ import annotations

from trading_ai.ea_generator.features_map import (
    EXPORT_SPEC, MQL5_HANDLE_CREATE, required_handles,
)
from trading_ai.ea_generator.mql_common import (
    array_1d, centroids_block, extract_model, model_constants,
)


def _handle_decls(handles: list[str]) -> str:
    """Dichiarazioni globali degli handle indicatore necessari."""
    return "\n".join(f"int h_{h} = INVALID_HANDLE;" for h in handles)


def _handle_init(handles: list[str]) -> str:
    """Codice OnInit che crea gli handle e ne verifica la validita'."""
    lines = []
    for h in handles:
        lines.append(f"   h_{h} = {MQL5_HANDLE_CREATE[h]};")
        lines.append(f"   if(h_{h} == INVALID_HANDLE) return(INIT_FAILED);")
    return "\n".join(lines)


def _feature_lines(feature_columns: list[str]) -> str:
    """Righe MQL5 che riempiono il vettore feature alla barra `s`."""
    lines = []
    for i, col in enumerate(feature_columns):
        expr = EXPORT_SPEC[col]["mql5"].format(s="s")
        lines.append(f"   f[{i}] = {expr};   // {col}")
    return "\n".join(lines)


def generate_mql5(strategy) -> str:
    """Ritorna il sorgente MQL5 completo della strategia data."""
    m = extract_model(strategy)
    risk = strategy.risk
    handles = required_handles(m["feature_columns"])

    return f"""//+------------------------------------------------------------------+
//|  {strategy.name}.mq5                                              |
//|  Expert Advisor generato automaticamente (Modulo 6 - EA Generator)|
//|  Pattern cluster {strategy.cluster_id}, direzione {'+1 BUY' if strategy.direction == 1 else '-1 SELL'} |
//+------------------------------------------------------------------+
#property strict
#include <Trade/Trade.mqh>      // classe CTrade per inviare/gestire ordini

{model_constants(m['n_features'], m['n_clusters'], strategy.cluster_id, strategy.direction)}
//--- Parametri di rischio ---
input double InpSL_ATR      = {risk.sl_atr};
input double InpTP_ATR      = {risk.tp_atr};
input double InpBE_ATR      = {risk.be_atr};
input double InpTrail_ATR   = {risk.trail_atr};
input double InpRiskPercent = {risk.risk_per_trade * 100.0};
input int    InpMaxBars     = {risk.max_bars};
input int    InpMaxTradesDay= {risk.max_trades_per_day};
input double InpMinADX      = 0.0;
input int    InpStartHour   = -1;
input int    InpEndHour     = -1;
input long   InpMagic       = {abs(hash(strategy.name)) % 90000 + 10000};

//--- Modello embeddato ---
{array_1d('gMean', m['mean'])}
{array_1d('gScale', m['scale'])}
{centroids_block('gCent', m['centroids'])}

//--- Handle indicatori ---
{_handle_decls(handles)}

CTrade   gTrade;            // helper per gli ordini
datetime gLastBar = 0;
int      gTradesToday = 0;
int      gDay = -1;

//+------------------------------------------------------------------+
int OnInit() {{
{_handle_init(handles)}
   gTrade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
}}
void OnDeinit(const int reason) {{}}

//+------------------------------------------------------------------+
//| Legge il valore di un buffer indicatore alla barra shift         |
//+------------------------------------------------------------------+
double IndVal(int handle, int buffer, int shift) {{
   double tmp[];
   if(CopyBuffer(handle, buffer, shift, 1, tmp) <= 0) return(0.0);
   return(tmp[0]);
}}

//+------------------------------------------------------------------+
void GetFeatures(int s, double &f[]) {{
{_feature_lines(m['feature_columns'])}
}}

void Standardize(double &f[]) {{
   for(int i = 0; i < FEAT_COUNT; i++) {{
      double sc = gScale[i];
      if(sc == 0.0) sc = 1.0;
      f[i] = (f[i] - gMean[i]) / sc;
   }}
}}

int NearestCluster(double &f[]) {{
   int best = -1; double bestDist = 0.0;
   for(int c = 0; c < N_CLUSTERS; c++) {{
      double d = 0.0;
      for(int j = 0; j < FEAT_COUNT; j++) {{
         double diff = f[j] - gCent[c * FEAT_COUNT + j];
         d += diff * diff;
      }}
      if(best < 0 || d < bestDist) {{ best = c; bestDist = d; }}
   }}
   return(best);
}}

//+------------------------------------------------------------------+
//| Posizione aperta di questo EA sul simbolo corrente?              |
//+------------------------------------------------------------------+
bool HasOpenPosition() {{
   if(PositionSelect(_Symbol))
      return(PositionGetInteger(POSITION_MAGIC) == InpMagic);
   return(false);
}}

//+------------------------------------------------------------------+
//| Lotto in base al rischio percentuale                             |
//+------------------------------------------------------------------+
double LotsByRisk(double slDistance) {{
   double riskMoney = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0;
   double tickVal  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double minLot   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(tickSize <= 0.0 || tickVal <= 0.0) return(minLot);
   double lossPerLot = slDistance / tickSize * tickVal;
   double lots = (lossPerLot > 0.0) ? riskMoney / lossPerLot : minLot;
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lots = MathFloor(lots / step) * step;
   lots = MathMax(lots, minLot);
   lots = MathMin(lots, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX));
   return(lots);
}}

//+------------------------------------------------------------------+
//| Gestione posizione aperta: BE, trailing, time-stop               |
//+------------------------------------------------------------------+
void ManageOpen() {{
   if(!PositionSelect(_Symbol)) return;
   if(PositionGetInteger(POSITION_MAGIC) != InpMagic) return;

   double atr = IndVal(h_atr_for_mgmt(), 0, 0);
   long   type = PositionGetInteger(POSITION_TYPE);
   double openP = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl    = PositionGetDouble(POSITION_SL);
   double tp    = PositionGetDouble(POSITION_TP);
   datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);

   if(InpMaxBars > 0 && (TimeCurrent() - openTime) >= InpMaxBars * PeriodSeconds()) {{
      gTrade.PositionClose(_Symbol);
      return;
   }}
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double newSL = sl;
   if(type == POSITION_TYPE_BUY) {{
      double profit = bid - openP;
      if(InpBE_ATR > 0 && profit >= InpBE_ATR * atr) newSL = MathMax(newSL, openP);
      if(InpTrail_ATR > 0) newSL = MathMax(newSL, bid - InpTrail_ATR * atr);
      if(newSL > sl + _Point) gTrade.PositionModify(_Symbol, newSL, tp);
   }} else if(type == POSITION_TYPE_SELL) {{
      double profit = openP - ask;
      if(InpBE_ATR > 0 && profit >= InpBE_ATR * atr)
         newSL = (sl == 0) ? openP : MathMin(newSL, openP);
      if(InpTrail_ATR > 0) {{
         double t = ask + InpTrail_ATR * atr;
         newSL = (sl == 0) ? t : MathMin(newSL, t);
      }}
      if(sl == 0 || newSL < sl - _Point) gTrade.PositionModify(_Symbol, newSL, tp);
   }}
}}

// ATR per la gestione: se l'handle 'atr' non e' tra le feature, ne creiamo uno dedicato.
int gAtrMgmt = INVALID_HANDLE;
int h_atr_for_mgmt() {{
   if(gAtrMgmt == INVALID_HANDLE) gAtrMgmt = iATR(_Symbol, _Period, 14);
   return(gAtrMgmt);
}}

bool FiltersOk() {{
   if(InpMinADX > 0) {{
      int ha = iADX(_Symbol, _Period, 14);
      if(IndVal(ha, 0, 1) < InpMinADX) return(false);
   }}
   if(InpStartHour >= 0 && InpEndHour >= 0) {{
      MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
      if(dt.hour < InpStartHour || dt.hour > InpEndHour) return(false);
   }}
   return(true);
}}

//+------------------------------------------------------------------+
void OnTick() {{
   datetime t0 = iTime(_Symbol, _Period, 0);
   if(t0 == gLastBar) return;            // una sola operazione per barra
   gLastBar = t0;

   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   if(dt.day != gDay) {{ gDay = dt.day; gTradesToday = 0; }}

   if(HasOpenPosition()) {{ ManageOpen(); return; }}
   if(InpMaxTradesDay > 0 && gTradesToday >= InpMaxTradesDay) return;
   if(!FiltersOk()) return;

   double f[FEAT_COUNT];
   GetFeatures(1, f);
   Standardize(f);
   if(NearestCluster(f) != TARGET_CLUSTER) return;

   double atr = IndVal(h_atr_for_mgmt(), 0, 1);
   if(atr <= 0) return;
   double slDist = InpSL_ATR * atr;
   double tpDist = InpTP_ATR * atr;
   double lots = LotsByRisk(slDist);

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(DIRECTION > 0) {{
      double sl = ask - slDist, tp = ask + tpDist;
      if(gTrade.Buy(lots, _Symbol, ask, sl, tp, "{strategy.name}")) gTradesToday++;
   }} else {{
      double sl = bid + slDist, tp = bid - tpDist;
      if(gTrade.Sell(lots, _Symbol, bid, sl, tp, "{strategy.name}")) gTradesToday++;
   }}
}}
//+------------------------------------------------------------------+
"""
