from datetime import datetime, timezone
import re
from typing import Optional, List
import random
import string

from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from bson.errors import InvalidId
from ..shared.settings import settings
from ..shared.auth_utils import current_user, verify_admin_access, verify_customer_service_access
from .models import (
    AvailableFlightIn, AvailableFlightOut, AvailableFlightUpdate,
    FlightPurchaseIn, FlightTicketOut, CustomerServiceAction
)
from .models import FlightCancelIn

app = FastAPI(title="Flight Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncIOMotorClient(settings.mongodb_uri)
db = client[settings.mongodb_db]
available_flights = db["available_flights"]  # Catálogo de voos disponíveis
purchased_flights = db["purchased_flights"]  # Tickets comprados pelos usuários
users = db["users"]
audit = db["audit_logs"]

def generate_seat_number():
    """Gera um número de assento aleatório"""
    row = random.randint(1, 30)
    seat = random.choice(['A', 'B', 'C', 'D', 'E', 'F'])
    return f"{row}{seat}"

# === ENDPOINTS PARA VOOS DISPONÍVEIS (CATÁLOGO) ===

@app.get("/flights/available", response_model=List[AvailableFlightOut])
async def list_available_flights(
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    date: Optional[str] = None,
    limit: int = 50
):
    """Listar voos disponíveis (público)"""
    query = {"status": "active"}
    
    if from_:
        # Matching por substring, case-insensitive
        # Escapar o input para evitar metacaracteres de regex
        pattern = re.escape(from_)
        query["from_"] = {"$regex": pattern, "$options": "i"}
    if to:
        query["to"] = to
    if date:
        query["date"] = {"$regex": date}  # Busca parcial por data
    
    cursor = available_flights.find(query).limit(limit)
    items = await cursor.to_list(length=None)
    
    result = []
    for flight in items:
        # Calcular assentos disponíveis (só conta tickets ativos)
        purchased_count = await purchased_flights.count_documents({
            "flightNumber": flight["flightNumber"],
            "status": "active"  # Só conta tickets ativos
        })
        available_seats = flight["totalSeats"] - purchased_count
        
        result.append(AvailableFlightOut(
            id=str(flight["_id"]),
            flightNumber=flight["flightNumber"],
            from_=flight["from_"],
            to=flight["to"],
            date=flight["date"],
            departureTime=flight["departureTime"],
            arrivalTime=flight["arrivalTime"],
            airline=flight["airline"],
            aircraft=flight["aircraft"],
            totalSeats=flight["totalSeats"],
            availableSeats=available_seats,
            price=flight["price"],
            status=flight["status"],
            createdAt=flight["createdAt"].isoformat(),
            updatedAt=flight["updatedAt"].isoformat(),
        ))
    
    return result

@app.get("/flights/available/{flight_id}", response_model=AvailableFlightOut)
async def get_available_flight(flight_id: str):
    """Buscar detalhes de um voo disponível"""
    flight = await available_flights.find_one({"_id": ObjectId(flight_id)})
    if not flight:
        raise HTTPException(status_code=404, detail="Flight not found")
    
    # Calcular assentos disponíveis (só tickets ativos ocupam assento)
    purchased_count = await purchased_flights.count_documents({
        "flightNumber": flight["flightNumber"],
        "status": "active"  # Cancelados e reembolsados não ocupam assento
    })
    available_seats = flight["totalSeats"] - purchased_count
    
    return AvailableFlightOut(
        id=str(flight["_id"]),
        flightNumber=flight["flightNumber"],
        from_=flight["from_"],
        to=flight["to"],
        date=flight["date"],
        departureTime=flight["departureTime"],
        arrivalTime=flight["arrivalTime"],
        airline=flight["airline"],
        aircraft=flight["aircraft"],
        totalSeats=flight["totalSeats"],
        availableSeats=available_seats,
        price=flight["price"],
        status=flight["status"],
        createdAt=flight["createdAt"].isoformat(),
        updatedAt=flight["updatedAt"].isoformat(),
    )

@app.post("/admin/flights/available", response_model=AvailableFlightOut)
async def create_available_flight(
    flight: AvailableFlightIn = Body(...),
    admin=Depends(verify_admin_access)
):
    """[ADMIN] Criar novo voo disponível no catálogo"""
    now = datetime.now(timezone.utc)
    
    # Verificar se já existe voo com mesmo número
    exists = await available_flights.find_one({"flightNumber": flight.flightNumber})
    if exists:
        raise HTTPException(status_code=409, detail="Flight number already exists")
    
    doc = {
        "flightNumber": flight.flightNumber,
        "from_": flight.from_,
        "to": flight.to,
        "date": flight.date,
        "departureTime": flight.departureTime,
        "arrivalTime": flight.arrivalTime,
        "airline": flight.airline,
        "aircraft": flight.aircraft,
        "totalSeats": flight.totalSeats,
        "price": flight.price,
        "status": flight.status,
        "createdAt": now,
        "updatedAt": now
    }
    
    res = await available_flights.insert_one(doc)
    
    # Log da criação
    await audit.insert_one({
        "userId": ObjectId(admin["_id"]),
        "action": "flight_created",
        "details": {
            "flightId": str(res.inserted_id),
            "flightNumber": flight.flightNumber
        },
        "createdAt": now
    })
    
    return AvailableFlightOut(
        id=str(res.inserted_id),
        flightNumber=doc["flightNumber"],
        from_=doc["from_"],
        to=doc["to"],
        date=doc["date"],
        departureTime=doc["departureTime"],
        arrivalTime=doc["arrivalTime"],
        airline=doc["airline"],
        aircraft=doc["aircraft"],
        totalSeats=doc["totalSeats"],
        availableSeats=doc["totalSeats"],  # Recém criado, todos disponíveis
        price=doc["price"],
        status=doc["status"],
        createdAt=doc["createdAt"].isoformat(),
        updatedAt=doc["updatedAt"].isoformat(),
    )

@app.put("/admin/flights/available/{flight_id}", response_model=AvailableFlightOut)
async def update_available_flight(
    flight_id: str,
    flight_update: AvailableFlightUpdate = Body(...),
    admin=Depends(verify_admin_access)
):
    """[ADMIN] Atualizar voo disponível"""
    flight = await available_flights.find_one({"_id": ObjectId(flight_id)})
    if not flight:
        raise HTTPException(status_code=404, detail="Flight not found")
    
    # Preparar campos para atualização (apenas campos não nulos)
    update_fields = {}
    for field, value in flight_update.dict(exclude_unset=True).items():
        if value is not None:
            update_fields[field] = value
    
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    update_fields["updatedAt"] = datetime.now(timezone.utc)
    
    await available_flights.update_one(
        {"_id": ObjectId(flight_id)},
        {"$set": update_fields}
    )
    
    # Log da atualização
    await audit.insert_one({
        "userId": ObjectId(admin["_id"]),
        "action": "flight_updated",
        "details": {
            "flightId": flight_id,
            "flightNumber": flight["flightNumber"],
            "updated_fields": list(update_fields.keys())
        },
        "createdAt": datetime.now(timezone.utc)
    })
    
    # Retornar voo atualizado
    updated_flight = await available_flights.find_one({"_id": ObjectId(flight_id)})
    
    # Calcular assentos disponíveis (só tickets ativos ocupam assento)
    purchased_count = await purchased_flights.count_documents({
        "flightNumber": updated_flight["flightNumber"],
        "status": "active"  # Cancelados e reembolsados liberam assento
    })
    available_seats = updated_flight["totalSeats"] - purchased_count
    
    return AvailableFlightOut(
        id=str(updated_flight["_id"]),
        flightNumber=updated_flight["flightNumber"],
        from_=updated_flight["from_"],
        to=updated_flight["to"],
        date=updated_flight["date"],
        departureTime=updated_flight["departureTime"],
        arrivalTime=updated_flight["arrivalTime"],
        airline=updated_flight["airline"],
        aircraft=updated_flight["aircraft"],
        totalSeats=updated_flight["totalSeats"],
        availableSeats=available_seats,
        price=updated_flight["price"],
        status=updated_flight["status"],
        createdAt=updated_flight["createdAt"].isoformat(),
        updatedAt=updated_flight["updatedAt"].isoformat(),
    )

# === ENDPOINTS PARA COMPRA DE VOOS (CLIENTES) ===

@app.post("/flights/purchase", response_model=FlightTicketOut)
async def purchase_flight(
    purchase: FlightPurchaseIn = Body(...),
    user=Depends(current_user)
):
    """Comprar um voo (apenas com número do voo)"""
    now = datetime.now(timezone.utc)
    
    # Buscar voo disponível
    available_flight = await available_flights.find_one({
        "flightNumber": purchase.flightNumber,
        "status": "active"
    })
    
    if not available_flight:
        raise HTTPException(status_code=404, detail="Flight not available")
    
    # Verificar se usuário já comprou este voo (só conta tickets ativos)
    exists = await purchased_flights.find_one({
        "userId": ObjectId(user["_id"]),
        "flightNumber": purchase.flightNumber,
        "status": "active"  # Só impede se tem ticket ativo
    })
    
    if exists:
        raise HTTPException(status_code=409, detail="You already have an active ticket for this flight")
    
    # Verificar disponibilidade de assentos (só conta tickets ativos)
    purchased_count = await purchased_flights.count_documents({
        "flightNumber": purchase.flightNumber,
        "status": "active"  # Cancelados e reembolsados liberam assento
    })
    
    if purchased_count >= available_flight["totalSeats"]:
        raise HTTPException(
            status_code=400, 
            detail=f"No seats available. {purchased_count}/{available_flight['totalSeats']} seats occupied"
        )
    
    # Criar ticket
    seat_number = generate_seat_number()
    doc = {
        "userId": ObjectId(user["_id"]),
        "flightNumber": purchase.flightNumber,
        "from_": available_flight["from_"],
        "to": available_flight["to"],
        "date": available_flight["date"],
        "departureTime": available_flight["departureTime"],
        "arrivalTime": available_flight["arrivalTime"],
        "airline": available_flight["airline"],
        "price": available_flight["price"],
        "seatNumber": seat_number,
        "status": "active",
        "createdAt": now,
        "updatedAt": now
    }
    
    res = await purchased_flights.insert_one(doc)
    
    # Log da compra
    await audit.insert_one({
        "userId": ObjectId(user["_id"]),
        "action": "flight_purchased",
        "details": {
            "ticketId": str(res.inserted_id),
            "flightNumber": purchase.flightNumber,
            "seatNumber": seat_number,
            "price": available_flight["price"]
        },
        "createdAt": now
    })
    
    return FlightTicketOut(
        id=str(res.inserted_id),
        userId=str(doc["userId"]),
        flightNumber=doc["flightNumber"],
        from_=doc["from_"],
        to=doc["to"],
        date=doc["date"],
        departureTime=doc["departureTime"],
        arrivalTime=doc["arrivalTime"],
        airline=doc["airline"],
        price=doc["price"],
        seatNumber=doc["seatNumber"],
        status=doc["status"],
        createdAt=doc["createdAt"].isoformat(),
        updatedAt=doc["updatedAt"].isoformat()
    )

@app.post("/flights/cancel/{ticket_id}")
async def cancel_ticket(ticket_id: str, user=Depends(current_user)):
    """Cancelar um ticket próprio"""
    res = await purchased_flights.update_one(
        {"_id": ObjectId(ticket_id), "userId": ObjectId(user["_id"]), "status": "active"},
        {"$set": {"status": "cancelled", "updatedAt": datetime.now(timezone.utc)}}
    )
    
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Ticket not found or already cancelled")
    
    # Log do cancelamento
    await audit.insert_one({
        "userId": ObjectId(user["_id"]),
        "action": "flight_cancellation",
        "details": {"ticketId": ticket_id},
        "createdAt": datetime.now(timezone.utc)
    })
    
    return {"detail": "Ticket cancelled successfully"}


@app.post("/flights/cancel-by-number")
async def cancel_ticket_by_number(payload: FlightCancelIn = Body(...), user=Depends(current_user)):
    """Cancelar o ticket ativo do usuário autenticado, informando apenas o número do voo.

    Busca um ticket ativo pertencente ao usuário para o `flightNumber` informado e o marca como `cancelled`.
    """
    flight_number = payload.flightNumber

    # Procurar ticket ativo do usuário para o flightNumber
    ticket = await purchased_flights.find_one({
        "userId": ObjectId(user["_id"]),
        "flightNumber": flight_number,
        "status": "active"
    })

    if not ticket:
        raise HTTPException(status_code=404, detail="Active ticket for this flight not found for current user")

    # Atualizar status para cancelled
    res = await purchased_flights.update_one(
        {"_id": ticket["_id"]},
        {"$set": {"status": "cancelled", "updatedAt": datetime.now(timezone.utc)}}
    )

    if res.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to cancel ticket")

    # Log do cancelamento
    await audit.insert_one({
        "userId": ObjectId(user["_id"]),
        "action": "flight_cancellation_by_number",
        "details": {
            "ticketId": str(ticket["_id"]),
            "flightNumber": flight_number,
            "reason": payload.reason
        },
        "createdAt": datetime.now(timezone.utc)
    })

    return {"detail": "Ticket cancelled successfully"}

@app.get("/flights/my-tickets", response_model=List[FlightTicketOut])
async def get_my_tickets(user=Depends(current_user)):
    """Listar meus tickets"""
    cursor = purchased_flights.find({"userId": ObjectId(user["_id"])})
    items = await cursor.to_list(length=None)
    
    result = []
    for ticket in items:
        result.append(FlightTicketOut(
            id=str(ticket["_id"]),
            userId=str(ticket["userId"]),
            flightNumber=ticket["flightNumber"],
            from_=ticket["from_"],
            to=ticket["to"],
            date=ticket["date"],
            departureTime=ticket["departureTime"],
            arrivalTime=ticket["arrivalTime"],
            airline=ticket["airline"],
            price=ticket["price"],
            seatNumber=ticket.get("seatNumber"),
            status=ticket["status"],
            createdAt=ticket["createdAt"].isoformat(),
            updatedAt=ticket["updatedAt"].isoformat(),
        ))
    
    return result

@app.get("/flights/my-tickets/active", response_model=List[FlightTicketOut])
async def get_my_active_tickets(user=Depends(current_user)):
    """Listar apenas meus tickets ativos"""
    cursor = purchased_flights.find({"userId": ObjectId(user["_id"]), "status": "active"})
    items = await cursor.to_list(length=None)

    result = []
    for ticket in items:
        result.append(FlightTicketOut(
            id=str(ticket["_id"]),
            userId=str(ticket["userId"]),
            flightNumber=ticket["flightNumber"],
            from_=ticket["from_"],
            to=ticket["to"],
            date=ticket["date"],
            departureTime=ticket["departureTime"],
            arrivalTime=ticket["arrivalTime"],
            airline=ticket["airline"],
            price=ticket["price"],
            seatNumber=ticket.get("seatNumber"),
            status=ticket["status"],
            createdAt=ticket["createdAt"].isoformat(),
            updatedAt=ticket["updatedAt"].isoformat(),
        ))

    return result

# === ENDPOINTS PARA CUSTOMER SERVICE (MCP) ===

@app.get("/cs/tickets/user/{user_id}", response_model=List[FlightTicketOut])
async def get_user_tickets_cs(user_id: str, cs_user=Depends(verify_customer_service_access)):
    """[CUSTOMER SERVICE] Listar tickets de um usuário específico"""
    try:
        uid = ObjectId(user_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid user id format")

    cursor = purchased_flights.find({"userId": uid})
    items = await cursor.to_list(length=None)
    
    result = []
    for ticket in items:
        result.append(FlightTicketOut(
            id=str(ticket["_id"]),
            userId=str(ticket["userId"]),
            flightNumber=ticket["flightNumber"],
            from_=ticket["from_"],
            to=ticket["to"],
            date=ticket["date"],
            departureTime=ticket["departureTime"],
            arrivalTime=ticket["arrivalTime"],
            airline=ticket["airline"],
            price=ticket["price"],
            seatNumber=ticket.get("seatNumber"),
            status=ticket["status"],
            createdAt=ticket["createdAt"].isoformat(),
            updatedAt=ticket["updatedAt"].isoformat(),
        ))
    
    return result

@app.get("/cs/tickets/user/{user_id}/active", response_model=List[FlightTicketOut])
async def get_user_active_tickets_cs(user_id: str, cs_user=Depends(verify_customer_service_access)):
    """[CUSTOMER SERVICE] Listar apenas tickets ativos de um usuário específico"""
    try:
        uid = ObjectId(user_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid user id format")

    cursor = purchased_flights.find({"userId": uid, "status": "active"})
    items = await cursor.to_list(length=None)

    result = []
    for ticket in items:
        result.append(FlightTicketOut(
            id=str(ticket["_id"]),
            userId=str(ticket["userId"]),
            flightNumber=ticket["flightNumber"],
            from_=ticket["from_"],
            to=ticket["to"],
            date=ticket["date"],
            departureTime=ticket["departureTime"],
            arrivalTime=ticket["arrivalTime"],
            airline=ticket["airline"],
            price=ticket["price"],
            seatNumber=ticket.get("seatNumber"),
            status=ticket["status"],
            createdAt=ticket["createdAt"].isoformat(),
            updatedAt=ticket["updatedAt"].isoformat(),
        ))

    return result

@app.get("/cs/tickets/user/{user_id}/all", response_model=List[FlightTicketOut])
async def get_user_all_tickets_cs(user_id: str, cs_user=Depends(verify_customer_service_access)):
    """Alias para listar todos os tickets de um usuário (compatibilidade)."""
    return await get_user_tickets_cs(user_id, cs_user)

@app.post("/cs/tickets/cancel/{ticket_id}")
async def cancel_ticket_cs(
    ticket_id: str,
    action: CustomerServiceAction = Body(...),
    cs_user=Depends(verify_customer_service_access)
):
    """[CUSTOMER SERVICE] Cancelar ticket de qualquer usuário"""
    try:
        oid = ObjectId(ticket_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid ticket id format")

    ticket = await purchased_flights.find_one({"_id": oid})
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    if ticket["status"] != "active":
        raise HTTPException(status_code=400, detail="Ticket is not active")
    
    res = await purchased_flights.update_one(
        {"_id": oid},
        {"$set": {"status": "cancelled", "updatedAt": datetime.now(timezone.utc)}}
    )
    
    # Log detalhado da ação do customer service
    await audit.insert_one({
        "userId": ticket["userId"],
        "action": "ticket_cancellation_cs",
        "performedBy": cs_user["_id"],
        "details": {
            "ticketId": ticket_id,
            "flightNumber": ticket["flightNumber"],
            "reason": action.reason,
            "performed_by": action.performed_by
        },
        "createdAt": datetime.now(timezone.utc)
    })
    
    return {"detail": "Ticket cancelled successfully by customer service"}

@app.post("/cs/tickets/refund/{ticket_id}")
async def refund_ticket_cs(
    ticket_id: str,
    action: CustomerServiceAction = Body(...),
    cs_user=Depends(verify_customer_service_access)
):
    """[CUSTOMER SERVICE] Processar reembolso de ticket"""
    try:
        oid = ObjectId(ticket_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid ticket id format")

    ticket = await purchased_flights.find_one({"_id": oid})
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    if ticket["status"] not in ["active", "cancelled"]:
        raise HTTPException(status_code=400, detail="Ticket cannot be refunded")
    
    res = await purchased_flights.update_one(
        {"_id": oid},
        {"$set": {"status": "refunded", "updatedAt": datetime.now(timezone.utc)}}
    )
    
    # Log do reembolso
    await audit.insert_one({
        "userId": ticket["userId"],
        "action": "ticket_refund_cs",
        "performedBy": cs_user["_id"],
        "details": {
            "ticketId": ticket_id,
            "flightNumber": ticket["flightNumber"],
            "reason": action.reason,
            "performed_by": action.performed_by
        },
        "createdAt": datetime.now(timezone.utc)
    })
    
    return {"detail": "Ticket refunded successfully"}

@app.get("/cs/tickets/search")
async def search_tickets_cs(
    flight_number: Optional[str] = None,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    cs_user=Depends(verify_customer_service_access)
):
    """[CUSTOMER SERVICE] Buscar tickets com filtros"""
    query = {}
    
    if flight_number:
        query["flightNumber"] = flight_number
    if user_id:
        try:
            query["userId"] = ObjectId(user_id)
        except (InvalidId, TypeError):
            raise HTTPException(status_code=400, detail="Invalid user id format")
    if status:
        query["status"] = status
    
    cursor = purchased_flights.find(query).limit(limit)
    items = await cursor.to_list(length=None)
    
    result = []
    for ticket in items:
        result.append(FlightTicketOut(
            id=str(ticket["_id"]),
            userId=str(ticket["userId"]),
            flightNumber=ticket["flightNumber"],
            from_=ticket["from_"],
            to=ticket["to"],
            date=ticket["date"],
            departureTime=ticket["departureTime"],
            arrivalTime=ticket["arrivalTime"],
            airline=ticket["airline"],
            price=ticket["price"],
            seatNumber=ticket.get("seatNumber"),
            status=ticket["status"],
            createdAt=ticket["createdAt"].isoformat(),
            updatedAt=ticket["updatedAt"].isoformat(),
        ))
    
    return result
