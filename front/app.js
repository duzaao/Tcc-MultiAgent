const form = document.getElementById('form');
const input = document.getElementById('inputMessage');
const messages = document.getElementById('messages');

function appendMessage(text, cls='bot'){
  const el = document.createElement('div');
  el.className = 'msg ' + (cls === 'user' ? 'user' : 'bot');
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

async function sendMessage(text){
  appendMessage(text, 'user');
  const body = { message: text};
  try{
    const resp = await fetch('/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await resp.json();
    if(data.error){
      appendMessage('Erro: ' + data.error);
    } else {
      appendMessage(data.reply || JSON.stringify(data));
      if(data.auto_confirmed){
        appendMessage('⚠️ Ação sensível confirmada automaticamente.');
      }
    }
  }catch(e){
    appendMessage('Erro de rede: ' + e.message);
  }
}

form.addEventListener('submit', (ev) =>{
  ev.preventDefault();
  const v = input.value && input.value.trim();
  if(!v) return;
  sendMessage(v);
  input.value = '';
});


appendMessage('Welcome! I am your assistant. How can I help you today? You can ask me questions or request actions like logging in or checking your flights.');
