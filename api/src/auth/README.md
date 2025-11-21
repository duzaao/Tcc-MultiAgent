# Authentication Service

Serviço responsável pelo gerenciamento de usuários, autenticação e autorização do sistema.

## Arquivos:

### `models.py`
- Modelos Pydantic para requisições e respostas
- `RegisterIn`, `LoginIn`, `TokenPairOut`, `MeOut`, `DeleteIn`

### `security.py` 
- Funções de segurança (hash de senhas, tokens JWT)
- Utiliza Argon2 para hash de senhas
- Geração e validação de tokens JWT

### `service.py`
- API FastAPI para autenticação
- Endpoints: register, login, refresh, logout, me, delete account
- Gerenciamento de sessões com refresh tokens

## Endpoints:

- `POST /auth/register` - Registrar novo usuário
- `POST /auth/login` - Fazer login e obter tokens
- `POST /auth/refresh` - Renovar token de acesso
- `POST /auth/logout` - Fazer logout (revoga refresh token)
- `GET /auth/me` - Obter informações do usuário atual
- `DELETE /auth/account` - Deletar conta do usuário

## Roles de Usuário:

- `customer` - Cliente comum (padrão)
- `customer_service` - Atendimento ao cliente (MCP)
- `admin` - Administrador do sistema

## Bancos de Dados:

- `users` - Dados dos usuários
- `sessions` - Sessões ativas (refresh tokens)
- `audit_logs` - Logs de auditoria

## Uso:

```bash
uvicorn src.auth.service:app --host 0.0.0.0 --port 8001
```
