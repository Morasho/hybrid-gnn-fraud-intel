import { useState } from 'react';
import axios from 'axios';
import { Signal, Wifi, Battery } from 'lucide-react';
import API_BASE from '../lib/api';

// ─── Screen state machine ────────────────────────────────────────────────────
const SCREENS = {
  HOME: 'HOME',
  MENU: 'MENU',
  ENTER_AMOUNT: 'ENTER_AMOUNT',
  CONFIRM: 'CONFIRM',
  PROCESSING: 'PROCESSING',
  RESULT: 'RESULT',
};

const currentTime = () =>
  new Date().toLocaleTimeString('en-KE', { hour: '2-digit', minute: '2-digit' });

const txId = () => 'MPE' + Math.random().toString(36).substr(2, 9).toUpperCase();

// ─── Sub-components ──────────────────────────────────────────────────────────

function StatusBar({ time }) {
  return (
    <div className="flex items-center justify-between px-4 py-1 text-white text-xs">
      <span className="font-medium">{time}</span>
      <div className="flex items-center gap-1">
        <Signal size={12} />
        <Wifi size={12} />
        <Battery size={12} />
      </div>
    </div>
  );
}

function USSDPanel({ title, lines = [], input, onInput, onSend, onBack, placeholder = '0', sending }) {
  return (
    <div className="p-4 h-full flex flex-col" style={{ fontFamily: 'monospace', color: '#00ff41' }}>
      <div className="text-xs mb-3 pb-2 border-b border-green-900 flex justify-between items-center">
        <span>Safaricom</span>
        <span className="text-green-600">*334#</span>
      </div>
      {title && <div className="text-sm font-bold mb-3 text-green-300">{title}</div>}
      <div className="flex-1 space-y-1 text-xs">
        {lines.map((line, i) => (
          <div key={i} className={line.startsWith('>') ? 'text-green-200' : 'text-green-400'}>
            {line}
          </div>
        ))}
      </div>
      {input !== undefined && (
        <div className="mt-3">
          <div className="border border-green-800 rounded px-2 py-1 flex items-center gap-2 bg-black/40">
            <span className="text-green-600 text-xs">›</span>
            <input
              autoFocus
              type="text"
              value={input}
              onChange={(e) => onInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && onSend()}
              placeholder={placeholder}
              className="flex-1 bg-transparent outline-none text-green-300 text-xs placeholder-green-900"
            />
          </div>
          <div className="flex gap-2 mt-2">
            {onBack && (
              <button
                onClick={onBack}
                className="flex-1 text-xs py-1 border border-green-900 rounded text-green-700 hover:text-green-400 transition"
              >
                Back
              </button>
            )}
            <button
              onClick={onSend}
              disabled={sending}
              className="flex-1 text-xs py-1 bg-green-900 border border-green-700 rounded text-green-300 hover:bg-green-800 transition disabled:opacity-50"
            >
              {sending ? 'Processing…' : 'Send'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function ResultScreen({ result, onReset }) {
  const blocked = result?.ResultCode === 1;
  const color = blocked ? '#ff4141' : '#00ff41';
  const title = blocked ? '⚠ TRANSACTION DECLINED' : '✓ TRANSACTION ACCEPTED';

  return (
    <div className="p-4 h-full flex flex-col" style={{ fontFamily: 'monospace' }}>
      <div className="text-xs mb-3 pb-2 border-b border-gray-800 flex justify-between" style={{ color: '#00ff41' }}>
        <span>Safaricom</span><span className="text-green-600">*334#</span>
      </div>
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center">
        <div className="text-3xl">{blocked ? '🚫' : '✅'}</div>
        <div className="text-sm font-bold" style={{ color }}>{title}</div>
        <div className="text-xs text-gray-400 leading-relaxed px-2">{result?.ResultDesc}</div>
        <div className="w-full mt-2 border border-gray-800 rounded p-3 text-left space-y-1">
          <div className="text-xs text-gray-600">Transaction ID</div>
          <div className="text-xs text-green-400 break-all">{result?.TransID}</div>
          <div className="text-xs text-gray-600 mt-2">AI Decision</div>
          <div className="text-xs font-bold" style={{ color }}>{result?.AIDecision}</div>
          <div className="text-xs text-gray-600 mt-2">Risk Score</div>
          <div className="text-xs" style={{ color }}>{result?.RiskScore}%</div>
        </div>
      </div>
      <button
        onClick={onReset}
        className="mt-4 w-full text-xs py-2 border border-green-900 rounded text-green-700 hover:text-green-400 hover:border-green-700 transition"
        style={{ fontFamily: 'monospace' }}
      >
        New Transaction (Back to Menu)
      </button>
    </div>
  );
}

// ─── Main exportable component ───────────────────────────────────────────────
// Props:
//   prefill  — optional { phone, amount } to pre-populate and jump to CONFIRM
//              (used by the analyst "Test Cases" panel)

export default function USSDPhoneUI({ prefill = null }) {
  const [screen, setScreen] = useState(
    prefill ? SCREENS.CONFIRM : SCREENS.HOME
  );
  const [phone, setPhone] = useState(prefill?.phone ?? '');
  const [amount, setAmount] = useState(prefill?.amount ?? '');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  const [time] = useState(currentTime);

  const reset = () => {
    setScreen(SCREENS.HOME);
    setPhone('');
    setAmount('');
    setResult(null);
    setError('');
  };

  const submitTransaction = async () => {
    const amt = parseFloat(amount);
    if (!phone.trim() || isNaN(amt) || amt <= 0) {
      setError('Phone and amount are required.');
      return;
    }
    setError('');
    setScreen(SCREENS.PROCESSING);
    setSending(true);

    const payload = {
      TransID: txId(),
      TransAmount: amt,
      MSISDN: phone.trim(),
      BillRefNumber: '00100',
      TransTime: new Date().toISOString(),
    };

    try {
      const res = await axios.post(`${API_BASE}/daraja-webhook`, payload);
      setResult(res.data);
    } catch {
      setResult({
        ResultCode: 1,
        ResultDesc: 'Could not reach Fraud Intel backend. Is the server running?',
        TransID: payload.TransID,
        AIDecision: 'BACKEND_OFFLINE',
        RiskScore: 0,
      });
    } finally {
      setSending(false);
      setScreen(SCREENS.RESULT);
    }
  };

  const renderScreen = () => {
    switch (screen) {
      case SCREENS.HOME:
        return (
          <USSDPanel
            title="M-Pesa"
            lines={['Welcome to M-Pesa', '', '1. Send Money', '2. Withdraw Cash', '3. Buy Airtime', '4. Pay Bill', '', 'Enter option:']}
            input=""
            onInput={() => {}}
            onSend={() => setScreen(SCREENS.MENU)}
            placeholder="1"
          />
        );
      case SCREENS.MENU:
        return (
          <USSDPanel
            title="Send Money"
            lines={['Enter recipient phone number.', '', 'Format: 0712345678', '        254712345678', '        +254712345678', '', 'All formats accepted.']}
            input={phone}
            onInput={setPhone}
            onSend={() => {
              if (!phone.trim()) { setError('Enter a phone number'); return; }
              setError('');
              setScreen(SCREENS.ENTER_AMOUNT);
            }}
            onBack={reset}
            placeholder="0712345678"
          />
        );
      case SCREENS.ENTER_AMOUNT:
        return (
          <USSDPanel
            title="Send Money"
            lines={[`To: ${phone}`, '', 'Enter amount (Ksh):', '', error ? `! ${error}` : '']}
            input={amount}
            onInput={setAmount}
            onSend={() => {
              const amt = parseFloat(amount);
              if (isNaN(amt) || amt <= 0) { setError('Enter a valid amount'); return; }
              setError('');
              setScreen(SCREENS.CONFIRM);
            }}
            onBack={() => setScreen(SCREENS.MENU)}
            placeholder="500"
          />
        );
      case SCREENS.CONFIRM:
        return (
          <USSDPanel
            title="Confirm Transaction"
            lines={[`Send Ksh ${parseFloat(amount || 0).toLocaleString()}`, `To: ${phone}`, '', 'The AI Fraud Engine will', 'screen this transaction', 'before processing.', '', 'Press Send to confirm.']}
            input=""
            onInput={() => {}}
            onSend={submitTransaction}
            onBack={() => setScreen(SCREENS.ENTER_AMOUNT)}
            placeholder=""
            sending={sending}
          />
        );
      case SCREENS.PROCESSING:
        return (
          <div className="p-4 h-full flex flex-col items-center justify-center gap-4" style={{ fontFamily: 'monospace', color: '#00ff41' }}>
            <div className="text-xs text-green-600 mb-2">Safaricom · *334#</div>
            <div className="text-2xl animate-pulse">🛡</div>
            <div className="text-sm text-green-300">Screening transaction…</div>
            <div className="text-xs text-green-700 text-center">Hybrid GNN + XGBoost<br />fraud engine running</div>
            <div className="flex gap-1 mt-2">
              {[0, 1, 2].map(i => (
                <div key={i} className="w-2 h-2 rounded-full bg-green-500 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
              ))}
            </div>
          </div>
        );
      case SCREENS.RESULT:
        return <ResultScreen result={result} onReset={reset} />;
      default:
        return null;
    }
  };

  return (
    <div className="relative mx-auto" style={{ width: 320 }}>
      <div className="bg-gray-900 rounded-[2.5rem] shadow-2xl border border-gray-700 p-3 pb-6">
        {/* Notch */}
        <div className="flex justify-center mb-1">
          <div className="w-20 h-5 bg-gray-800 rounded-full flex items-center justify-center gap-1">
            <div className="w-2 h-2 rounded-full bg-gray-600" />
            <div className="w-1 h-1 rounded-full bg-gray-500" />
          </div>
        </div>
        <StatusBar time={time} />
        {/* Screen */}
        <div className="mx-1 rounded-2xl overflow-hidden" style={{ background: '#0a0a12', minHeight: 480 }}>
          {renderScreen()}
        </div>
        {/* Home bar */}
        <div className="flex justify-center mt-3">
          <div className="w-24 h-1 bg-gray-600 rounded-full" />
        </div>
      </div>
    </div>
  );
}
