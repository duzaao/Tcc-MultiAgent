# Flight Service

Serviço responsável pelo gerenciamento de voos, compras de passagens e operações de customer service.

## Arquivos:

### `models.py`
- Modelos Pydantic para voos e tickets
- `AvailableFlightIn/Out`, `FlightPurchaseIn`, `FlightTicketOut`, `CustomerServiceAction`

### `service.py`
- API FastAPI para operações de voos
- Endpoints para catálogo público, compras e customer service

## Funcionalidades:

### 🌐 **Catálogo Público de Voos**
- Listar voos disponíveis com filtros (origem, destino, data)
- Ver detalhes de voos específicos
- Cálculo automático de assentos disponíveis

### 🛒 **Sistema de Compras**
- Compra de passagens com verificação de disponibilidade
- Prevenção de overbooking
- Geração automática de assentos
- Controle de compras duplicadas por usuário

### 🔧 **Operações de Customer Service (MCP)**
- Buscar tickets por usuário, voo ou status
- Cancelar tickets de qualquer usuário
- Processar reembolsos
- Logs detalhados de todas as ações

### 👨‍💼 **Administração**
- Criar novos voos no catálogo
- Atualizar informações de voos existentes
- Controle total sobre o catálogo

## Endpoints:

### Público:
- `GET /flights/available` - Listar voos disponíveis
- `GET /flights/available/{id}` - Detalhes de um voo

### Cliente Autenticado:
- `POST /flights/purchase` - Comprar passagem
- `POST /flights/cancel/{ticket_id}` - Cancelar própria passagem
- `GET /flights/my-tickets` - Listar minhas passagens

### Customer Service:
- `GET /cs/tickets/user/{user_id}` - Tickets de um usuário
- `POST /cs/tickets/cancel/{ticket_id}` - Cancelar ticket
- `POST /cs/tickets/refund/{ticket_id}` - Processar reembolso
- `GET /cs/tickets/search` - Buscar tickets

### Admin:
- `POST /admin/flights/available` - Criar voo
- `PUT /admin/flights/available/{id}` - Atualizar voo

## Lógica de Assentos:

- ✅ **Assentos ocupados**: Apenas tickets com status "active"
- ❌ **Assentos liberados**: Tickets "cancelled" ou "refunded"
- 🚫 **Prevenção de overbooking**: Verificação automática antes da compra
- 🎲 **Geração de assento**: Aleatório (1A-30F)

## Bancos de Dados:

- `available_flights` - Catálogo de voos disponíveis
- `purchased_flights` - Tickets comprados pelos usuários
- `audit_logs` - Logs de todas as operações

## Uso:

```bash
uvicorn src.flights.service:app --host 0.0.0.0 --port 8002
```
