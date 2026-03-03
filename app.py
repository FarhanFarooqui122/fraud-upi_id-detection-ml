import pandas as pd
import re
import math
import joblib
from collections import Counter
from difflib import SequenceMatcher

# ============================================================
# CONSTANTS — must match train.py exactly
# ============================================================

LEGIT_HANDLES = list(set([
    'oksbi', 'sbi', 'sbipay',
    'okhdfcbank', 'hdfc', 'hdfcbank',
    'okicici', 'icici', 'icicipay',
    'okaxis', 'axl', 'axis',
    'kotak', 'kmbl', 'kotakbank',
    'paytm', 'ptaxis', 'pthdfc', 'ptsbi',
    'ybl', 'ibl',
    'apl', 'yapl',
    'upi', 'npci',
    'pnb', 'boi', 'bob', 'cnrb', 'union',
    'indus', 'idbi', 'federal', 'hsbc', 'citi',
    'scb', 'rbl', 'yes', 'idfc', 'dbs',
    'cub', 'kvb', 'iob', 'centralbank',
    'syndicatebank', 'allahabad', 'andb',
    'corporation', 'vijayabank', 'denabank',
    'orientalbank', 'uco', 'fbl', 'ikwik',
    'pingpay', 'jupiteraxis', 'slice', 'niyoicici',
    'fifederal', 'naviaxis', 'mahb', 'jkb',
    'karnataka', 'tmbl', 'dcb', 'equitas',
    'suryoday', 'utib', 'ratn', 'ibkl',
    'barodampay', 'cmsidfc','ptys', 'ptsbi', 'ptaxis', 'pthdfc', 'ptyes',
'ptkotak', 'ptidfc', 'ptindus', 'ptrbl', 'ptfederal',
]))

LEGIT_HANDLES_SET = set(LEGIT_HANDLES)

TRUSTED_VPAS = [
    'sbi@oksbi', 'hdfc@okhdfcbank', 'paytm@paytm',
    'npci@upi', 'icici@okicici'
]

SUSPICIOUS_KEYWORDS = list(set([
    'kyc', 'ekyc', 'kycupdate', 'kycverify', 'kycexpired',
    'verify', 'verification', 'verified', 'unverified',
    'validate', 'validation', 'activate', 'activation',
    'secure', 'security', 'secured', 'unsecure',
    'blocked', 'unblock', 'suspend', 'suspended',
    'locked', 'unlock', 'freeze', 'frozen',
    'disabled', 'enable', 'deactivate', 'reactivate',
    'bank', 'banking', 'netbanking', 'mobilebanking',
    'account', 'savings', 'current', 'deposit',
    'withdrawal', 'transfer', 'transaction', 'payment',
    'balance', 'statement', 'passbook', 'cheque',
    'loan', 'emi', 'credit', 'debit',
    'otp', 'pin', 'mpin', 'password', 'passwd',
    'credential', 'login', 'signin', 'signup',
    'auth', 'authenticate', 'token', 'session',
    'help', 'helpdesk', 'helpline', 'helpcenter',
    'support', 'customersupport', 'customercare',
    'service', 'services', 'care', 'assist',
    'tollfree', 'tollfreenumber', 'callcenter',
    'grievance', 'complaint', 'feedback', 'query',
    'contact', 'reach', 'connect', 'chat',
    'official', 'offical', 'govt', 'gov', 'government',
    'india', 'bharat', 'national', 'central', 'state',
    'ministry', 'department', 'authority', 'board',
    'commission', 'corporation', 'council', 'bureau',
    'rbi', 'npci', 'sebi', 'irdai', 'pfrda',
    'uidai', 'gstn', 'incometax', 'cbdt', 'cbic',
    'enforcement', 'cbi', 'ed', 'itr',
    'tax', 'gst', 'tds', 'refund',
    'taxrefund', 'itrrefund', 'taxreturn', 'taxfiling',
    'taxnotice', 'taxpay', 'taxdue', 'taxclear',
    'reward', 'rewards', 'bonus', 'cashback',
    'offer', 'offers', 'deal', 'deals', 'discount',
    'coupon', 'voucher', 'gift', 'giftcard',
    'free', 'freebie', 'freemoney', 'freerecharge',
    'lucky', 'luckydraw', 'winner', 'winning',
    'prize', 'prizewin', 'jackpot', 'lottery',
    'congratulations', 'congrats', 'selected', 'chosen',
    'urgent', 'urgently', 'immediately', 'asap',
    'deadline', 'lastchance', 'expire', 'expiry',
    'warning', 'alert', 'notice', 'notification',
    'actionrequired', 'required', 'mandatory',
    'final', 'lastnotice', 'critical',
    'money', 'cash', 'fund', 'funds', 'finance',
    'wallet', 'paywallet', 'moneywallet', 'ewallet',
    'earn', 'earnings', 'income', 'salary', 'wage',
    'profit', 'investment', 'invest',
    'scheme', 'schemes', 'program', 'programme',
    'yojana', 'pradhanmantri', 'pm', 'pmjay',
    'pmkisan', 'pmawas', 'pmjandhan', 'ujjwala',
    'ayushman', 'mudra', 'startup', 'digital',
    'modi', 'pmo', 'cmo',
    'president', 'minister', 'collector', 'officer',
    'inspector', 'police', 'magistrate', 'judge',
    'crypto', 'bitcoin', 'btc', 'eth', 'usdt',
    'nft', 'mining', 'miner',
    'trade', 'trading', 'forex', 'stock', 'share',
    'loans', 'personalloan', 'homeloan',
    'carloan', 'instant', 'quickloan', 'easyloan',
    'approval', 'approved', 'sanctioned', 'disburse',
    'recharge', 'topup', 'bill', 'billpay',
    'electricity', 'water', 'gas', 'dth',
    'test', 'demo', 'sample', 'dummy', 'fake',
    'temp', 'temporary', 'trial', 'pilot',
    'admin', 'root', 'superuser', 'master',
    'backup', 'recovery', 'restore', 'reset',
]))


# ============================================================
# FEATURE EXTRACTION — same as train.py
# ============================================================

def extract_features(vpa):
    vpa = vpa.lower().strip()

    if '@' not in vpa:
        return None

    local, handle = vpa.split('@', 1)

    features = {}

    features['vpa_length'] = len(vpa)
    features['local_length'] = len(local)
    features['handle_length'] = len(handle)

    features['digit_count'] = sum(c.isdigit() for c in local)
    features['digit_ratio'] = features['digit_count'] / max(len(local), 1)
    features['special_char_count'] = sum(c in '._-' for c in local)
    features['uppercase_count'] = sum(c.isupper() for c in vpa)
    features['has_hyphen_in_local'] = int('-' in local)
    features['has_official_keyword'] = int(
        any(kw in local for kw in ['npci', 'rbi', 'care', 'govt', 'official', 'support'])
    )
    features['is_qr_pattern'] = int(bool(re.match(r'^(paytmqr|gpay|phonepe|bhimupi)[a-z0-9]{6,10}$', local)))

    features['handle_is_known'] = int(handle in LEGIT_HANDLES_SET)
    features['handle_has_digits'] = int(bool(re.search(r'\d', handle)))
    features['handle_has_hyphen'] = int('-' in handle)

    local_parts = re.split(r'[.\-_]', local)
    features['has_suspicious_keyword'] = int(
        any(kw in part for part in local_parts for kw in SUSPICIOUS_KEYWORDS)
    )
    features['suspicious_part_count'] = sum(
        1 for part in local_parts for kw in SUSPICIOUS_KEYWORDS if kw in part
    )

    features['local_is_all_digits'] = int(local.isdigit())
    features['local_very_long'] = int(len(local) > 20)
    features['local_very_short'] = int(len(local) < 3)
    features['consecutive_digits'] = len(max(re.findall(r'\d+', local) or [''], key=len))

    max_sim = 0
    for trusted in TRUSTED_VPAS:
        sim = SequenceMatcher(None, vpa, trusted).ratio()
        max_sim = max(max_sim, sim)
    features['max_trusted_similarity'] = max_sim
    features['is_close_to_trusted'] = int(0.7 < max_sim < 1.0)

    counts = Counter(local)
    entropy = -sum((c / len(local)) * math.log2(c / len(local))
                   for c in counts.values())
    features['local_entropy'] = round(entropy, 4)

    return features


# ============================================================
# PREDICTION
# ============================================================

def predict_upi_fraud(upi_id):
    model = joblib.load('upi_fraud_model.pkl')
    columns = joblib.load('feature_columns.pkl')

    if '@' not in upi_id:
        return {"upi_id": upi_id, "risk_score": 100,
                "risk_level": "HIGH 🚨", "action": "Block - Invalid format",
                "top_signal": "missing_@"}

    local, handle = upi_id.split('@', 1)

    if len(local) < 3:
        return {"upi_id": upi_id, "risk_score": 85,
                "risk_level": "HIGH 🚨", "action": "Block - Too short",
                "top_signal": "local_very_short"}

    features = extract_features(upi_id)
    if not features:
        return {"upi_id": upi_id, "risk_score": 100,
                "risk_level": "HIGH 🚨", "action": "Block - Feature extraction failed",
                "top_signal": "invalid_format"}

    try:
        X_input = pd.DataFrame([features])[columns]
        fraud_prob = model.predict_proba(X_input)[0][1]
        risk_score = round(fraud_prob * 100, 2)

        if risk_score < 30:
            risk_level = "LOW ✅"
            action = "Allow"
        elif risk_score < 65:
            risk_level = "MEDIUM ⚠️"
            action = "Request additional verification"
        else:
            risk_level = "HIGH 🚨"
            action = "Block transaction"

        feature_importance = dict(zip(columns, model.feature_importances_))
        top_feature = max(feature_importance, key=feature_importance.get)

        return {
            "upi_id": upi_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "action": action,
            "top_signal": top_feature
        }

    except Exception as e:
        return {"upi_id": upi_id, "risk_score": 100,
                "risk_level": "HIGH 🚨", "action": f"Block - Error: {str(e)}",
                "top_signal": "exception"}


# ============================================================
# CLI
# ============================================================

print("\n" + "="*50)
print("   UPI FRAUD DETECTION SYSTEM")
print("="*50)

while True:
    print("\nEnter a UPI ID to check (or type 'exit' to quit):")
    upi_input = input(">>> ").strip()

    if upi_input.lower() == 'exit':
        print("Exiting... Goodbye!")
        break

    if upi_input == "":
        print("Please enter a valid UPI ID.")
        continue

    result = predict_upi_fraud(upi_input)

    print("\n" + "-"*40)
    print(f"  UPI ID     : {result['upi_id']}")
    print(f"  Risk Score : {result['risk_score']}")
    print(f"  Risk Level : {result['risk_level']}")
    print(f"  Action     : {result['action']}")
    print(f"  Top Signal : {result['top_signal']}")
    print("-"*40)