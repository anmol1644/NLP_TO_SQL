// This file is no longer needed since we're using direct API calls to the backend now.
// CORS issues are fixed directly on the backend API.
// We're keeping this file just for reference.

import { NextRequest, NextResponse } from 'next/server';

// Return a simple message if someone tries to access this route
export async function GET(request: NextRequest) {
  return new NextResponse(
    JSON.stringify({ message: "This proxy is disabled. Please use direct API calls to the backend." }),
    {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }
  );
}

export async function POST(request: NextRequest) {
  return new NextResponse(
    JSON.stringify({ message: "This proxy is disabled. Please use direct API calls to the backend." }),
    {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }
  );
}

export async function PUT(request: NextRequest) {
  return new NextResponse(
    JSON.stringify({ message: "This proxy is disabled. Please use direct API calls to the backend." }),
    {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }
  );
}

export async function DELETE(request: NextRequest) {
  return new NextResponse(
    JSON.stringify({ message: "This proxy is disabled. Please use direct API calls to the backend." }),
    {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }
  );
}

export async function OPTIONS(request: NextRequest) {
  return new NextResponse(
    JSON.stringify({ message: "This proxy is disabled. Please use direct API calls to the backend." }),
    {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }
  );
} 