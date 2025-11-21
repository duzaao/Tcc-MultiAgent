# Shared Components

Esta pasta contém componentes compartilhados entre todos os serviços do sistema.

## Arquivos:

### `settings.py`
- Configurações globais do sistema
- Variáveis de ambiente
- Configurações do MongoDB, JWT, CORS

### `auth_utils.py`
- Funções de autenticação compartilhadas
- Validação de tokens JWT
- Verificação de permissões (admin, customer service)
- Cliente MongoDB compartilhado

## Uso:

```python
from src.shared.settings import settings
from src.shared.auth_utils import current_user, verify_admin_access
```

Estes componentes são usados por:
- Serviço de Autenticação
- Serviço de Voos
- Servidor MCP
