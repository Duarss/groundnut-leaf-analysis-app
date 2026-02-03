// src/utils/clientId.js
export function getClientId() {
  const key = "groundnut_client_id";
  let cid = localStorage.getItem(key);
  if (!cid) {
    cid = crypto.randomUUID();
    localStorage.setItem(key, cid);
  }
  return cid;
}
