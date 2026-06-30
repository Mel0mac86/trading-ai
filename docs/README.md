# AI Trading — app PWA (chat con l'AI della piattaforma)

App installabile su iPhone/Android per **chattare liberamente con l'AI di
`trading-ai`**, senza passare da Claude Code. Funziona via browser e si installa
come un'app dalla schermata Home.

## Come attivarla (una volta sola)

1. Su GitHub: **Settings → Pages**.
2. In *Build and deployment* → *Source* scegli **Deploy from a branch**.
3. Branch: il branch su cui vive questa cartella (es. `main`), cartella **`/docs`**.
   Salva.
4. Dopo ~1 minuto l'app è online a:
   `https://mel0mac86.github.io/trading-ai/`

## Come installarla su iPhone

1. Apri quell'indirizzo in **Safari**.
2. Tocca il pulsante **Condividi** → **Aggiungi a Home**.
3. Si crea l'icona "AI Trading": si apre a tutto schermo come un'app.

## Prima chat: serve la chiave API

L'app parla con Claude usando **la tua chiave API Anthropic**:

1. Creane una su <https://console.anthropic.com/settings/keys> (serve credito).
2. Aprila app → ⚙︎ **Impostazioni** → incolla la chiave → **Salva**.

🔒 **Privacy**: la chiave e la cronologia restano **solo sul tuo telefono**
(memoria locale del browser). Non finiscono mai su GitHub né su altri server:
le richieste vanno direttamente dal browser ad `api.anthropic.com`. Il repository
è pubblico, quindi **nessuna chiave è scritta nel codice**.

⚠️ Se hai incollato una chiave in chat o in un posto condiviso, **revocala** e
creane una nuova.

## Cosa sa fare

L'AI è già "istruita" sulla piattaforma: i 9 moduli, la strategia robusta
**PAT17_SHORT** su XAUUSD H1, i numeri reali (e quali sono artefatti), e come
migliorare la ricerca su Kaggle. Puoi chiederle spiegazioni, rischi, prossimi
passi. Non dà garanzie di guadagno: il trading comporta rischio di perdita.
