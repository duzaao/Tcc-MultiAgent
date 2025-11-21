"""
MCP Server para Customer Service - Flight Management
Versão simplificada usando FastMCP
"""

import asyncio
import json
import httpx
import os
from mcp.server.fastmcp import FastMCP

# Configurações
FLIGHT_SERVICE_URL = os.getenv("FLIGHT_SERVICE_URL", "http://localhost:8002")
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
# Nota: não manteremos token em variável global; usar get_customer_token() para leitura dinâmica

# Token storage (persistência mínima em tokens.env)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENS_FILE = os.path.join(PROJECT_ROOT, "tokens.env")


def _write_tokens_env(data: dict):
    """Escreve/atualiza chaves no arquivo tokens.env (simples, preserva outras linhas)."""
    try:
        lines = []
        if os.path.exists(TOKENS_FILE):
            with open(TOKENS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" not in line:
                        lines.append(line.rstrip("\n"))
                        continue
                    k = line.split("=", 1)[0]
                    if k in data:
                        continue
                    lines.append(line.rstrip("\n"))
        for k, v in data.items():
            lines.append(f"{k}={v}")
        with open(TOKENS_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def set_customer_token(token: str):
    """Persiste token em tokens.env e no ambiente do processo."""
    try:
        # Persistir em arquivo
        _write_tokens_env({"CUSTOMER_SERVICE_TOKEN": token})
        # Também expor no ambiente do processo para conveniência
        os.environ["CUSTOMER_SERVICE_TOKEN"] = token
    except Exception:
        pass


def get_customer_token() -> str:
    """Retorna token atual: primeiro verifica variável de ambiente, depois tokens.env."""

    if os.path.exists(TOKENS_FILE):
        try:
            with open(TOKENS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("CUSTOMER_SERVICE_TOKEN="):
                        return line.split("=", 1)[1].strip()
        except Exception:
            # On any error reading the file, return empty string to indicate no valid token.
            return ""

    # No token found
    return ""


# Criar aplicação MCP
app = FastMCP("Flight Customer Service")

@app.tool()
async def search_customer_flights(user_id: str) -> str:
    """
    description:
        Retrieve all flights for a specific customer, including cancelled and refunded.

    args:
        user_id (str): The customer identifier in the system.

    returns:
        str: JSON string containing the list of flights or an error message.

    common_phrases:
        - "find flights for user"
        - "list all reservations"
        - "customer flights by id"
    """
    try:
        async with httpx.AsyncClient() as client:
            token = get_customer_token()
            if not token:
                return (
                    "❌ No token configured. Please run the login tool first.\n"
                    "Usage example: login <username> <password>"
                )
            headers = {"Authorization": f"Bearer {token}"}
            url = f"{FLIGHT_SERVICE_URL}/cs/tickets/user/{user_id}/all"
            
            response = await client.get(url, headers=headers)
            
            if response.status_code == 200:
                flights = response.json()
                return f"✅ Found {len(flights)} flights for user {user_id} (all records):\n" + json.dumps(flights, indent=2, ensure_ascii=False)
            else:
                return f"❌ Error fetching flights: {response.status_code} - {response.text}"
                
    except Exception as e:
        return f"❌ Erro na busca: {str(e)}"

@app.tool()
async def change_password(username: str, email: str, new_password: str) -> str:
    """
    description:
        Change a user's password given username, email and the new password.

    args:
        username (str): Account username.
        email (str): Registered email address for verification.
        new_password (str): The new password to set.

    returns:
        str: Success or error message.

    common_phrases:
        - "change password"
        - "reset password"
        - "update user credentials"
    """
    try:
        async with httpx.AsyncClient() as client:
            url = f"{AUTH_SERVICE_URL}/auth/change-password"
            headers = {"Content-Type": "application/json"}
            payload = {"username": username, "email": email, "new_password": new_password}
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return "✅ Password changed successfully."
            else:
                return f"❌ Error changing password: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Erro na alteração de senha: {str(e)}"
    
@app.tool()
async def list_available_flights(from_: str = None) -> str:
    """
    description:
        List available flights for purchase, optionally filtered by origin.

    args:
        from_ (str, optional): Origin code or name to filter flights. Defaults to None.

    returns:
        str: JSON string of available flights or an error message.

    common_phrases:
        - "available flights"
        - "flights from"
        - "list flights for sale"
    """
    try:
        async with httpx.AsyncClient() as client:
            url = f"{FLIGHT_SERVICE_URL}/flights/available"
            params = None
            if from_:
                params = {"from": from_}

            # Aumentar timeout para evitar ReadTimeout em chamadas lentas
            try:
                response = await client.get(url, params=params, timeout=10.0)
            except httpx.ReadTimeout:
                return "❌ Erro na listagem: timeout ao contactar o Flight Service (leitura). Tente novamente mais tarde."

            if response.status_code == 200:
                flights = response.json()
                if from_:
                    return f"✅ Available flights filtered by origin '{from_}' ({len(flights)}):\n" + json.dumps(flights, indent=2, ensure_ascii=False)
                return f"✅ Available flights ({len(flights)}):\n" + json.dumps(flights, indent=2, ensure_ascii=False)
            else:
                return f"❌ Error listing flights: {response.status_code} - {response.text}"

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Retornar traceback completo para facilitar debug no MCP
        return f"❌ Erro na listagem: {str(e)}\n{tb}"

@app.tool()
async def get_flight_details(flight_id: str) -> str:
    """
    description:
        Get full details for a specific flight by its identifier.

    args:
        flight_id (str): Identifier of the flight.

    returns:
        str: JSON string with flight details or an error message.

    common_phrases:
        - "flight details"
        - "get flight info"
        - "flight by id"
    """
    try:
        async with httpx.AsyncClient() as client:
            url = f"{FLIGHT_SERVICE_URL}/flights/available/{flight_id}"
            
            response = await client.get(url)
            
            if response.status_code == 200:
                flight = response.json()
                return f"✅ Flight {flight_id} details:\n" + json.dumps(flight, indent=2, ensure_ascii=False)
            else:
                return f"❌ Error getting details: {response.status_code} - {response.text}"
                
    except Exception as e:
        return f"❌ Erro na consulta: {str(e)}"

@app.tool()
async def search_flights_by_status(status: str) -> str:
    """
    description:
        Search flights or tickets filtered by status (e.g., active, cancelled, refunded).

    args:
        status (str): Status to filter by, such as 'active', 'cancelled', or 'refunded'.

    returns:
        str: JSON string with filtered flights or an error message.

    common_phrases:
        - "flights by status"
        - "search cancelled flights"
        - "active tickets"
    """
    try:
        async with httpx.AsyncClient() as client:
            token = get_customer_token()
            if not token:
                return (
                    "❌ No token configured. Please run the login tool first.\n"
                    "Usage example: login <username> <password>"
                )
            headers = {"Authorization": f"Bearer {token}"}
            url = f"{FLIGHT_SERVICE_URL}/cs/tickets/search"
            params = {"status": status}
            
            response = await client.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                flights = response.json()
                return f"✅ Flights with status '{status}' ({len(flights)}):\n" + json.dumps(flights, indent=2, ensure_ascii=False)
            else:
                return f"❌ Error searching flights: {response.status_code} - {response.text}"
                
    except Exception as e:
        return f"❌ Erro na consulta: {str(e)}"

@app.tool()
async def cancel_customer_flight(ticket_id: str, reason: str, performed_by: str = "", executed_by: str = "mcp") -> str:
    """
    description:
        Cancel a customer's flight ticket on behalf of the user and release the seat.

    args:
        ticket_id (str): Ticket identifier to cancel.
        reason (str): Reason for cancellation.
        performed_by (str, optional): Name of the requester. Defaults to empty.
        executed_by (str, optional): Name of the agent performing the action. Default 'mcp'.

    returns:
        str: JSON result of the cancellation or an error message.

    common_phrases:
        - "cancel ticket"
        - "customer cancellation"
        - "release seat"
    """
    try:
        async with httpx.AsyncClient() as client:
            # Garantir que o campo performed_by seja preenchido — se não informado, usar executed_by para compatibilidade
            final_performed_by = performed_by or executed_by

            token = get_customer_token()
            if not token:
                return "❌ Nenhum token configurado. Execute login(username, password) primeiro."
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            url = f"{FLIGHT_SERVICE_URL}/cs/tickets/cancel/{ticket_id}"
            data = {
                "reason": reason,
                "performed_by": final_performed_by,
                "executed_by": executed_by,
            }

            response = await client.post(url, headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                return f"✅ Flight cancelled successfully!\n" + json.dumps(result, indent=2, ensure_ascii=False)
            else:
                return f"❌ Error cancelling flight: {response.status_code} - {response.text}"

    except Exception as e:
        return f"❌ Erro no cancelamento: {str(e)}"

@app.tool()
async def process_flight_refund(ticket_id: str, reason: str, executed_by: str, performed_by: str = "") -> str:
    """
    description:
        Process a refund for a flight ticket and free the associated seat.

    args:
        ticket_id (str): Ticket identifier to refund.
        reason (str): Reason for the refund.
        executed_by (str): Name of the actor executing the refund.
        performed_by (str, optional): Name of the requester. Defaults to empty.

    returns:
        str: JSON result of the refund operation or an error message.

    common_phrases:
        - "process refund"
        - "ticket refund"
        - "refund request"
    """
    try:
        async with httpx.AsyncClient() as client:
            token = get_customer_token()
            if not token:
                return "❌ Nenhum token configurado. Execute login(username, password) primeiro."
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            url = f"{FLIGHT_SERVICE_URL}/cs/tickets/refund/{ticket_id}"
            # Garantir compatibilidade com o Customer Service, que exige 'performed_by'
            final_performed_by = performed_by or executed_by
            data = {
                "reason": reason,
                "performed_by": final_performed_by,
                "executed_by": executed_by,
            }
            
            response = await client.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return f"✅ Refund processed successfully!\n" + json.dumps(result, indent=2, ensure_ascii=False)
            else:
                return f"❌ Error processing refund: {response.status_code} - {response.text}"
                
    except Exception as e:
        return f"❌ Erro no reembolso: {str(e)}"

@app.tool()
async def search_flights_by_number(flight_number: str) -> str:
    """
    description:
        List all tickets related to a specific flight number.

    args:
        flight_number (str): Flight number string, e.g. 'BR2024'.

    returns:
        str: JSON string with tickets or an error message.

    common_phrases:
        - "tickets for flight"
        - "search by flight number"
        - "flight tickets list"
    """
    try:
        async with httpx.AsyncClient() as client:
            token = get_customer_token()
            if not token:
                return "❌ Nenhum token configurado. Execute login(username, password) primeiro."
            headers = {"Authorization": f"Bearer {token}"}
            url = f"{FLIGHT_SERVICE_URL}/cs/tickets/search"
            params = {"flight_number": flight_number}
            
            response = await client.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                tickets = response.json()
                return f"✅ Tickets for flight {flight_number} ({len(tickets)}):\n" + json.dumps(tickets, indent=2, ensure_ascii=False)
            else:
                return f"❌ Error searching tickets: {response.status_code} - {response.text}"
                
    except Exception as e:
        return f"❌ Erro na consulta: {str(e)}"

@app.tool()
async def login(username: str, password: str) -> str:
    """
    description:
        Authenticate with the Auth Service using username and password and store the returned token.

    args:
        username (str): Account username.
        password (str): Account password.

    returns:
        str: Success message with a saved token preview, or an error message.

    common_phrases:
        - "login"
        - "authenticate user"
        - "save token"
    """
    try:
        async with httpx.AsyncClient() as client:
            url = f"{AUTH_SERVICE_URL}/auth/login"
            response = await client.post(url, json={"username": username, "password": password})
            if response.status_code in (200, 201):
                payload = response.json()
                token = payload.get("access_token") or payload.get("token") or payload.get("accessToken")
                if token:
                    set_customer_token(token)
                    return f"✅ Login successful. Token saved (preview): {token[:20]}..."
                else:
                    return f"✅ Login completed but no token was returned: {json.dumps(payload, ensure_ascii=False)}"

            # Helpful suggestions on common failure modes
            if response.status_code in (401, 403):
                return (
                    f"❌ Login failed: unauthorized ({response.status_code}). "
                    "Check your username and password.\n" 
                    "If you forgot your password, consider running change_password <username> <email> <new_password> or use the Auth service's reset flow.\n"
                    "Usage example: login <username> <password>"
                )

            if response.status_code == 400:
                # If the server signals a bad request, suggest correct usage
                return (
                    f"❌ Bad request when trying to login: {response.status_code} - {response.text}.\n"
                    "Make sure you provide both username and password.\n"
                    "Usage example: login <username> <password>"
                )

            return f"❌ Login failed: {response.status_code} - {response.text}"
    except httpx.RequestError as e:
        # Network/DNS errors: provide actionable hint for Docker vs host setups
        hint = (
            "If you're running the MCP server inside Docker, 'localhost' refers to the container. "
            "Make sure AUTH_SERVICE_URL points to the API container hostname (for docker-compose use 'http://api:8001') "
            "or to 'http://host.docker.internal:8001' on Docker Desktop."
        )
        return f"❌ Login error (network): {str(e)}. AUTH_SERVICE_URL={AUTH_SERVICE_URL}. {hint}"
    except Exception as e:
        return f"❌ Login error: {str(e)}"

@app.tool()
async def register(username: str, password: str, email: str = None, role: str = "customer") -> str:
    """
    description:
        Register a new user in the Auth Service with username, password, optional email and role.

    args:
        username (str): Desired username.
        password (str): Desired password.
        email (str, optional): User email. Defaults to None.
        role (str, optional): User role. Defaults to 'customer'.

    returns:
        str: Success or error message from registration.

    common_phrases:
        - "register user"
        - "create account"
        - "signup"
    """
    try:
        async with httpx.AsyncClient() as client:
            url = f"{AUTH_SERVICE_URL}/auth/register"
            payload = {"username": username, "password": password, "role": role}
            if email:
                payload["email"] = email

            response = await client.post(url, json=payload)

            if response.status_code in (200, 201):
                return f"✅ Registration successful: {username}"
            else:
                return f"❌ Registration failed: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Registration error: {str(e)}"


def _remove_customer_token():
    """Remove token do arquivo tokens.env e da memória."""
    # Remover do ambiente do processo
    try:
        if "CUSTOMER_SERVICE_TOKEN" in os.environ:
            os.environ.pop("CUSTOMER_SERVICE_TOKEN")
    except Exception:
        pass

    try:
        if os.path.exists(TOKENS_FILE):
            lines = []
            with open(TOKENS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.startswith("CUSTOMER_SERVICE_TOKEN="):
                        lines.append(line.rstrip("\n"))
            with open(TOKENS_FILE, "w", encoding="utf-8") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")
                else:
                    f.write("")
    except Exception:
        pass

@app.tool()
async def logout() -> str:
    """
    description:
        Log out locally by removing the saved customer token from storage and environment.

    args:
        None

    returns:
        str: Confirmation message or an error message.

    common_phrases:
        - "logout"
        - "remove token"
        - "sign out"
    """
    try:
        _remove_customer_token()
        return "✅ Logout complete. Local token removed."
    except Exception as e:
        return f"❌ Logout error: {str(e)}"


@app.tool()
async def buy_flight(flight_number: str = None, flight_id: str = None, seat_class: str = None, payment_token: str = None) -> str:
    """
    description:
        Purchase a flight using flight number (or flight_id) and the authenticated user's token.

    args:
        flight_number (str, optional): Flight number to purchase. Can be provided as flight_id alias.
        flight_id (str, optional): Alias for flight_number for compatibility.
        seat_class (str, optional): Requested seat class.
        payment_token (str, optional): Payment token or reference.

    returns:
        str: Purchase confirmation JSON or an error message.

    common_phrases:
        - "buy flight"
        - "purchase ticket"
        - "checkout flight"
    """
    # Compatibilidade: aceitar 'flight_id' como alias de 'flight_number'
    if not flight_number and flight_id:
        flight_number = flight_id

    if not flight_number:
        return "❌ No flight number provided. Please supply 'flight_number' or 'flight_id'."

    token = get_customer_token()
    if not token:
        return (
            "❌ No token configured. Please run the login tool first.\n"
            "Usage example: login <username> <password>"
        )

    try:
        async with httpx.AsyncClient() as client:
            purchase_url = f"{FLIGHT_SERVICE_URL}/flights/purchase"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            data = {"flightNumber": flight_number}
            if seat_class:
                data["seat_class"] = seat_class
            if payment_token:
                data["payment_token"] = payment_token

            response = await client.post(purchase_url, headers=headers, json=data)

            if response.status_code in (200, 201):
                try:
                    ticket = response.json()
                    return "✅ Purchase successful:\n" + json.dumps(ticket, indent=2, ensure_ascii=False)
                except Exception:
                    return "✅ Purchase successful (non-JSON response):\n" + response.text

            if response.status_code in (401, 403):
                return (
                    "❌ Unauthorized (401/403). Token is invalid or expired. Please run login <username> <password> again."
                )

            if response.status_code == 404:
                return "❌ Endpoint /flights/purchase not found (404). Verify the Flights service is running."

            return f"❌ Purchase error: {response.status_code} - {response.text}"

    except httpx.RequestError as e:
        return f"❌ Erro de rede ao tentar comprar: {str(e)}"
    except Exception as e:
        return f"❌ Erro inesperado na compra: {str(e)}"

#@app.tool()
#async def search_customer_active_flights(user_id: str) -> str:
#    """
#    Busca apenas os tickets ATIVOS de um cliente específico (Customer Service).
#
#    Chama '/cs/tickets/user/{user_id}/active' — requer token de Customer Service.
#    """
#    try:
#        async with httpx.AsyncClient() as client:
#            token = get_customer_token()
#            if not token:
#                return "❌ Nenhum token configurado. Execute login(username, password) primeiro."
#            headers = {"Authorization": f"Bearer {token}"}
#            url = f"{FLIGHT_SERVICE_URL}/cs/tickets/user/{user_id}/active"
#
#            response = await client.get(url, headers=headers)
#
#            if response.status_code == 200:
#                flights = response.json()
#                return f"✅ Encontrados {len(flights)} tickets ativos para o usuário {user_id}:\n" + json.dumps(flights, indent=2, ensure_ascii=False)
#            else:
#                return f"❌ Erro ao buscar tickets ativos: {response.status_code} - {response.text}"
#    except Exception as e:
#        return f"❌ Erro na busca (ativos): {str(e)}"


@app.tool()
async def my_active_tickets() -> str:
    """
    description:
        List the authenticated user's active tickets using the user's token.

    args:
        None

    returns:
        str: JSON string with active tickets or an error message.

    common_phrases:
        - "my active tickets"
        - "list my tickets"
        - "user tickets"
    """
    token = get_customer_token()
    if not token:
        return (
            "❌ No user token configured. Please run login <username> <password> as the purchasing user first."
        )

    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {token}"}
            url = f"{FLIGHT_SERVICE_URL}/flights/my-tickets/active"

            response = await client.get(url, headers=headers)

            if response.status_code == 200:
                tickets = response.json()
                return f"✅ Your active tickets ({len(tickets)}):\n" + json.dumps(tickets, indent=2, ensure_ascii=False)
            elif response.status_code in (401, 403):
                return "❌ Unauthorized. User token invalid/expired. Please run login <username> <password> again."
            else:
                return f"❌ Error listing your active tickets: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Erro na consulta dos seus tickets ativos: {str(e)}"



@app.tool()
async def list_my_purchased_flights() -> str:
    """
    description:
        Alias for listing the authenticated user's active purchased tickets.

    args:
        None

    returns:
        str: Same output as `my_active_tickets`, a JSON string or error message.

    common_phrases:
        - "my purchased flights"
        - "list purchased tickets"
        - "user purchases"
    """
    # delegar para a implementação existente
    return await my_active_tickets()


@app.tool()
async def list_my_flights() -> str:
    """
    description:
        Friendly alias to list the authenticated user's active flights/tickets.

    args:
        None

    returns:
        str: Same output as `my_active_tickets`, a JSON string or error message.

    common_phrases:
        - "list my flights"
        - "my flights"
        - "user tickets"
    """
    return await my_active_tickets()


@app.tool()
async def list_my_tickets() -> str:
    """
    description:
        Alternative alias to list the authenticated user's tickets.

    args:
        None

    returns:
        str: Same output as `my_active_tickets`, a JSON string or error message.

    common_phrases:
        - "my tickets"
        - "list tickets"
        - "active tickets"
    """
    return await my_active_tickets()


@app.tool()
async def listar_meus_voos() -> str:
    """
    description:
        Portuguese-language alias to list the authenticated user's active flights.

    args:
        None

    returns:
        str: Same output as `my_active_tickets`, a JSON string or error message.

    common_phrases:
        - "listar meus voos"
        - "meus voos"
        - "tickets ativos"
    """
    return await my_active_tickets()


@app.tool()
async def meus_voos() -> str:
    """
    description:
        Short Portuguese alias to list the authenticated user's flights/tickets.

    args:
        None

    returns:
        str: Same output as `my_active_tickets`, a JSON string or error message.

    common_phrases:
        - "meus voos"
        - "listar voos"
        - "tickets do usuario"
    """
    return await my_active_tickets()


@app.tool()
async def cancel_my_flight_by_number(flight_number: str, reason: str = "") -> str:
    """
    description:
        Cancel the authenticated user's active ticket by providing the flight number.

    args:
        flight_number (str): Flight number of the ticket to cancel.
        reason (str, optional): Cancellation reason. Defaults to empty.

    returns:
        str: Confirmation message or an error message from the Flight Service.

    common_phrases:
        - "cancel my flight"
        - "cancel by flight number"
        - "user cancel flight"
    """
    token = get_customer_token()
    if not token:
        return "❌ No user token configured. Please run login <username> <password> first."

    try:
        async with httpx.AsyncClient() as client:
            url = f"{FLIGHT_SERVICE_URL}/flights/cancel-by-number"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            payload = {"flightNumber": flight_number, "reason": reason}

            response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                return f"✅ Flight {flight_number} cancelled successfully for the authenticated user."
            if response.status_code in (401, 403):
                return "❌ Unauthorized. Token invalid/expired. Please run login <username> <password> again."
            if response.status_code == 404:
                return f"❌ No active ticket found for flight {flight_number} for the authenticated user."

            return f"❌ Error cancelling flight: {response.status_code} - {response.text}"

    except Exception as e:
        return f"❌ Erro no cancelamento do voo: {str(e)}"

if __name__ == "__main__":
    # Verificar se o token está configurado
    token = get_customer_token()

    print("🚀 Iniciando MCP Server - Flight Customer Service")
    print(f"🔐 Token configurado: {token[:20]}...")
    print(f"✈️ Flight Service: {FLIGHT_SERVICE_URL}")
    print(f"🔑 Auth Service: {AUTH_SERVICE_URL}")
    print("📡 Servidor MCP pronto para conexões!")
    
    # Executar servidor
    app.run()
