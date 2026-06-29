"""
Generatore di Expert Advisor MQL4 (.mq4) - Modulo 6.

Produce un EA autosufficiente che:
  - ricalcola le stesse feature native su cui e' addestrato il modello,
  - le standardizza coi parametri embeddati dello scaler,
  - assegna la barra al cluster piu' vicino (KMeans embeddato),
  - se il cluster coincide con quello del pattern, apre un trade nella direzione
    della strategia con SL/TP/Break-Even/Trailing in multipli di ATR e
    dimensionamento a rischio percentuale.

NB: questo file GENERA testo MQL4; la compilazione va fatta in MetaEditor. Il
codice segue le API standard MQL4 ed e' fortemente commentato.
"""

from __future__ import annotations

from trading_ai.ea_generator.features_map import EXPORT_SPEC
from trading_ai.ea_generator.mql_common import (
    array_1d, centroids_block, extract_model, model_constants,
)


def _feature_lines(feature_columns: list[str]) -> str:
    """Genera le righe MQL4 che riempiono il vettore feature alla barra `s`."""
    lines = []
    for i, col in enumerate(feature_columns):
        expr = EXPORT_SPEC[col]["mql4"].format(s="s")
        lines.append(f"   f[{i}] = {expr};   // {col}")
    return "\n".join(lines)


def generate_mql4(strategy) -> str:
    """Ritorna il sorgente MQL4 completo della strategia data."""
    m = extract_model(strategy)
    risk = strategy.risk
    feat_lines = _feature_lines(m["feature_columns"])

    return f"""//+------------------------------------------------------------------+
//|  {strategy.name}.mq4                                              |
//|  Expert Advisor generato automaticamente (Modulo 6 - EA Generator)|
//|  Pattern cluster {strategy.cluster_id}, direzione {'+1 BUY' if strategy.direction == 1 else '-1 SELL'} |
//+------------------------------------------------------------------+
#property strict

{model_constants(m['n_features'], m['n_clusters'], strategy.cluster_id, strategy.direction)}
//--- Parametri di rischio (modificabili dall'utente nel terminale) ---
input double InpSL_ATR      = {risk.sl_atr};   // Stop Loss in multipli di ATR
input double InpTP_ATR      = {risk.tp_atr};   // Take Profit in multipli di ATR
input double InpBE_ATR      = {risk.be_atr};   // Break-even dopo N*ATR di profitto (0=off)
input double InpTrail_ATR   = {risk.trail_atr};// Trailing stop in N*ATR (0=off)
input double InpRiskPercent = {risk.risk_per_trade * 100.0};   // % di equity rischiata per trade
input int    InpMaxBars     = {risk.max_bars}; // Time-stop: chiusura dopo N barre
input int    InpMaxTradesDay= {risk.max_trades_per_day};       // Max trade al giorno (0=illimitato)
input double InpMinADX      = 0.0;             // Filtro: ADX minimo (0=off)
input int    InpStartHour   = -1;              // Filtro orario inizio (-1=off)
input int    InpEndHour     = -1;              // Filtro orario fine
input int    InpMagic       = {abs(hash(strategy.name)) % 90000 + 10000};        // Magic number univoco

//--- Modello embeddato (scaler + centroidi KMeans) ---
{array_1d('gMean', m['mean'])}
{array_1d('gScale', m['scale'])}
{centroids_block('gCent', m['centroids'])}

datetime gLastBar = 0;     // timestamp dell'ultima barra processata
int      gTradesToday = 0; // contatore trade del giorno
int      gDay = -1;        // giorno corrente (per reset)

//+------------------------------------------------------------------+
int OnInit() {{ return(INIT_SUCCEEDED); }}
void OnDeinit(const int reason) {{}}

//+------------------------------------------------------------------+
//| Riempie il vettore feature alla barra di shift s                 |
//+------------------------------------------------------------------+
void GetFeatures(int s, double &f[]) {{
{feat_lines}
}}

//+------------------------------------------------------------------+
//| Standardizza in-place: (x - media) / scala                       |
//+------------------------------------------------------------------+
void Standardize(double &f[]) {{
   for(int i = 0; i < FEAT_COUNT; i++) {{
      double sc = gScale[i];
      if(sc == 0.0) sc = 1.0;                 // evita divisione per zero
      f[i] = (f[i] - gMean[i]) / sc;
   }}
}}

//+------------------------------------------------------------------+
//| Indice del centroide piu' vicino (distanza euclidea al quadrato) |
//+------------------------------------------------------------------+
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
   return best;
}}

//+------------------------------------------------------------------+
//| Esiste gia' una posizione aperta di questo EA?                   |
//+------------------------------------------------------------------+
bool HasOpenPosition() {{
   for(int i = OrdersTotal() - 1; i >= 0; i--) {{
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         if(OrderMagicNumber() == InpMagic && OrderSymbol() == _Symbol)
            return(true);
   }}
   return(false);
}}

//+------------------------------------------------------------------+
//| Dimensionamento del lotto in base al rischio percentuale         |
//+------------------------------------------------------------------+
double LotsByRisk(double slDistance) {{
   double riskMoney = AccountBalance() * InpRiskPercent / 100.0;
   double tickVal  = MarketInfo(_Symbol, MODE_TICKVALUE);
   double tickSize = MarketInfo(_Symbol, MODE_TICKSIZE);
   if(tickSize <= 0.0 || tickVal <= 0.0) return(MarketInfo(_Symbol, MODE_MINLOT));
   double lossPerLot = slDistance / tickSize * tickVal;     // perdita di 1 lotto allo SL
   double lots = (lossPerLot > 0.0) ? riskMoney / lossPerLot : MarketInfo(_Symbol, MODE_MINLOT);
   double step = MarketInfo(_Symbol, MODE_LOTSTEP);
   lots = MathFloor(lots / step) * step;                    // arrotonda allo step
   lots = MathMax(lots, MarketInfo(_Symbol, MODE_MINLOT));
   lots = MathMin(lots, MarketInfo(_Symbol, MODE_MAXLOT));
   return(lots);
}}

//+------------------------------------------------------------------+
//| Gestione di una posizione aperta: BE, trailing, time-stop        |
//+------------------------------------------------------------------+
void ManageOpen() {{
   double atr = iATR(_Symbol, _Period, 14, 0);
   for(int i = OrdersTotal() - 1; i >= 0; i--) {{
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderMagicNumber() != InpMagic || OrderSymbol() != _Symbol) continue;

      // Time-stop: chiusura se la posizione e' troppo vecchia.
      if(InpMaxBars > 0 && (TimeCurrent() - OrderOpenTime()) >= InpMaxBars * PeriodSeconds()) {{
         double px = (OrderType() == OP_BUY) ? Bid : Ask;
         OrderClose(OrderTicket(), OrderLots(), px, 5);
         continue;
      }}
      double newSL = OrderStopLoss();
      if(OrderType() == OP_BUY) {{
         double profit = Bid - OrderOpenPrice();
         if(InpBE_ATR > 0 && profit >= InpBE_ATR * atr)            // break-even
            newSL = MathMax(newSL, OrderOpenPrice());
         if(InpTrail_ATR > 0)                                      // trailing
            newSL = MathMax(newSL, Bid - InpTrail_ATR * atr);
         if(newSL > OrderStopLoss() + Point)
            OrderModify(OrderTicket(), OrderOpenPrice(), newSL, OrderTakeProfit(), 0);
      }} else if(OrderType() == OP_SELL) {{
         double profit = OrderOpenPrice() - Ask;
         if(InpBE_ATR > 0 && profit >= InpBE_ATR * atr)
            newSL = (newSL == 0) ? OrderOpenPrice() : MathMin(newSL, OrderOpenPrice());
         if(InpTrail_ATR > 0) {{
            double t = Ask + InpTrail_ATR * atr;
            newSL = (newSL == 0) ? t : MathMin(newSL, t);
         }}
         if(newSL < OrderStopLoss() - Point || OrderStopLoss() == 0)
            OrderModify(OrderTicket(), OrderOpenPrice(), newSL, OrderTakeProfit(), 0);
      }}
   }}
}}

//+------------------------------------------------------------------+
//| Filtri di contesto: ritorna true se l'ingresso e' consentito     |
//+------------------------------------------------------------------+
bool FiltersOk() {{
   if(InpMinADX > 0 && iADX(_Symbol, _Period, 14, PRICE_CLOSE, MODE_MAIN, 1) < InpMinADX)
      return(false);
   if(InpStartHour >= 0 && InpEndHour >= 0) {{
      int h = TimeHour(TimeCurrent());
      if(h < InpStartHour || h > InpEndHour) return(false);
   }}
   return(true);
}}

//+------------------------------------------------------------------+
void OnTick() {{
   // Operiamo una sola volta per barra (sulla barra appena chiusa, shift 1).
   if(Time[0] == gLastBar) return;
   gLastBar = Time[0];

   // Reset contatore giornaliero.
   int today = TimeDay(TimeCurrent());
   if(today != gDay) {{ gDay = today; gTradesToday = 0; }}

   if(HasOpenPosition()) {{ ManageOpen(); return; }}   // niente pyramiding

   if(InpMaxTradesDay > 0 && gTradesToday >= InpMaxTradesDay) return;
   if(!FiltersOk()) return;

   // Calcolo feature sulla barra chiusa, standardizzazione, cluster.
   double f[FEAT_COUNT];
   GetFeatures(1, f);
   Standardize(f);
   if(NearestCluster(f) != TARGET_CLUSTER) return;

   double atr = iATR(_Symbol, _Period, 14, 1);
   if(atr <= 0) return;
   double slDist = InpSL_ATR * atr;
   double tpDist = InpTP_ATR * atr;
   double lots = LotsByRisk(slDist);

   if(DIRECTION > 0) {{                                  // ---- BUY ----
      double price = Ask;
      double sl = price - slDist;
      double tp = price + tpDist;
      if(OrderSend(_Symbol, OP_BUY, lots, price, 5, sl, tp,
                   "{strategy.name}", InpMagic, 0, clrBlue) > 0)
         gTradesToday++;
   }} else {{                                            // ---- SELL ----
      double price = Bid;
      double sl = price + slDist;
      double tp = price - tpDist;
      if(OrderSend(_Symbol, OP_SELL, lots, price, 5, sl, tp,
                   "{strategy.name}", InpMagic, 0, clrRed) > 0)
         gTradesToday++;
   }}
}}
//+------------------------------------------------------------------+
"""
